import sys
import os
import codecs
import locale
import subprocess

os.chdir(os.getcwd())
sys.path.insert(0, '../Functions_JZ')
import Save_Load_File
import Pd_Funs

# 　input of this class is filepath of Tenoke_ccta_dl_segment_label.exe and location of where CTA.nii.gz
# output of this class is to extract centerline file based on cta file
# now this program support up to 2 threads, otherwise, there will be too much load on cpu
import threading
import time
from PySide2.QtUiTools import QUiLoader

try:
    import psutil
except ImportError:  # Keep the legacy module importable in minimal environments.
    psutil = None

class CenterlineEx:
    def __init__(self, UI=None, Hedys=None, exePath='', folderPath=None, TablePath=None):
        self.ui = None
        if UI:
            self.ui = UI
        self.modelui = None
        if Hedys:
            self.modelui = Hedys.ui

        self.ui.chooseExeBtn_CG_1.clicked.connect(lambda: self.ChooseExeFile())
        # used for tab 2 choose exe file
        self.ui.ChooseExeBtn_CG_3.clicked.connect(lambda: self.ChooseExeFile(flag=1))
        self.ui.chooseSaveBtn_CG_1.clicked.connect(lambda: self.ChooseCTADir())
        self.ui.ChooseExcelPathBtn_CG_3.clicked.connect(lambda: self.ChooseTablePath())
        self.ui.GenerateBtn_CG_1.clicked.connect(lambda: self.iterate())
        self.ui.BatchProcessBtn_CG_3.clicked.connect(lambda: self.BatchViaExcel())

        self.InitCenterlineEx(exePath=exePath, folderPath=folderPath, TablePath=TablePath)

    def InitCenterlineEx(self, exePath='', folderPath=None, TablePath=None):
        self.destination = folderPath
        self.root_address = ''
        self.part1_address = exePath
        self.part105_address = ''
        self.part2_address = ''
        self.part3_address = ''
        self.part4_address = ''
        self.pathCollection = []
        self.count = 0
        self.thread1Collection = []
        self.thread2Collection = []
        self.DataFrame = None
        self.ManualFlag = False
        self.CSVPath = TablePath

    # to choose the path of part1 exe file
    def ChooseExeFile(self, flag=0):
        ExePath = Save_Load_File.OpenFilePathQt(
            dispMsg="Load Exe Path",
            fileTypes='All files (*.*);; exe files(*.exe)',
            fileObj=self.ui,
            qtObj=True
        )
        if flag == 0:
            self.ui.LoadExePathTxt_CG_1.setPlainText(ExePath)
        else:
            self.ui.LoadExePathTxt_CG_3.setPlainText(ExePath)

    def ChooseTablePath(self):
        TablePath = Save_Load_File.OpenFilePathQt(
            dispMsg='Choose Excel File Path',
            fileTypes='All files (*.*);; excel files(*.csv *.xlsx)',
            fileObj=self.ui,
            qtObj=True
        )

        self.ui.TablePathTxt_CG_3.setPlainText(TablePath)

    # used to choose the directory path
    def ChooseCTADir(self):
        DirName = Save_Load_File.OpenDirPathQt(
            dispMsg="Save directory",
            fileObj=self.ui,
            qtObj=True
        )

        self.ui.LoadCTAPathTxt_CG_1.setPlainText(DirName)

    # use the file path of part1 exe file to set the path of other exe files
    def rearrange(self):
        print(self.part1_address)

        if r'/' in self.part1_address:
            self.root_address = '/'.join(self.part1_address.split('/')[:-2])
        else:
            self.root_address = '/'.join(self.part1_address.split('\\')[:-2])
        self.root_address += '/'
        # print('root', self.root_address)

        self.part105_address = self.root_address + r'Tenoke_ccta_Part2Alt\Tenoke_ccta_Part2Alt.exe'
        self.part2_address = self.root_address + r'Tenoke_ccta_Part2AltB\Tenoke_ccta_Part2AltB.exe'
        self.part3_address = self.root_address + r'Tenoke_ccta_Part3\Tenoke_ccta_Part3.exe'
        self.part4_address = self.root_address + r'Tenoke_ccta_Part4\Tenoke_ccta_Part4.exe'
    # iterate the chosen directory to get the absolute path of CTA file
    def iterate(self):
        # 　use \\ to replace / in the path
        for (root, dirs, files) in os.walk(self.ui.LoadCTAPathTxt_CG_1.toPlainText().replace(r'/', '\\')):
            if 'CTA.nii.gz' in files:
                self.pathCollection.append(root)
            else:
                pass

        self.part1_address = self.ui.LoadExePathTxt_CG_1.toPlainText()
        self.rearrange()

        # to check if the multi-thread method is chosen
        if not self.ui.MultiThreadCBox_CG_1.isChecked():
            for path in self.pathCollection:
                self.destination = path
                # self.DetectManual()
                self.run()

        # if multi-thread method is chosen, rearrange the
        else:
            # split the list of path contain cta file into two lists
            split_list = self.split_list_average_n(self.pathCollection, 2)
            self.thread1Collection = next(split_list)
            self.thread2Collection = next(split_list)

            print('thread1', self.thread1Collection)
            print('thread2', self.thread2Collection)

            thread1 = myThread(threadID=1, name='thread-1', folderPath=self.thread1Collection,
                               exePath=self.part1_address)
            thread2 = myThread(threadID=2, name='thread-2', folderPath=self.thread2Collection,
                               exePath=self.part1_address)
            thread1.start()
            thread2.start()

        # clear the path in storage
        self.initialize()

    # divide list into n equal lists
    def split_list_average_n(self, origin_list, n):
        if len(origin_list) % n == 0:
            cnt = len(origin_list) // n
        else:
            cnt = len(origin_list) // n + 1

        for i in range(0, n):
            yield origin_list[i * cnt:(i + 1) * cnt]

    # read the chosen excel file and convert it to dataframe
    def ReadExcelInfo(self):
        if self.CSVPath:
            self.DataFrame = Pd_Funs.OpenDF(self.CSVPath, header=0)
        else:
            # breakpoint()
            self.DataFrame = Pd_Funs.OpenDF(self.ui.TablePathTxt_CG_3.toPlainText(), header=0)
        # print(self.DataFrame['FolderPath'])

    # used to detect if manual tracing file exists
    def DetectManual(self):
        manualFile = 'Combo_Vals_Man.nii.gz'
        maunalPath = os.path.join(self.destination, manualFile)
        if os.path.exists(maunalPath):
            self.ManualFlag = True
        else:
            self.ManualFlag = False

    def BatchViaExcel(self):
        if not self.part1_address:
            self.part1_address = self.ui.LoadExePathTxt_CG_3.toPlainText()
        self.ReadExcelInfo()

        for i in range(len(self.DataFrame)):
            self.destination = self.DataFrame['FolderPath'][i]
            self.DetectManual()
            self.run()

    # this method used to empty all the storage
    def initialize(self):
        self.thread1Collection = []
        self.thread2Collection = []
        self.pathCollection = []

    @staticmethod
    def _safe_stage_commands(exe_path, destination, manual_flag):
        """Return the vendor pipeline as argument lists, without using a shell."""

        exe_path = os.path.abspath(str(exe_path))
        destination = os.path.abspath(str(destination))
        executable_root = os.path.dirname(os.path.dirname(exe_path))

        commands = [
            ("part1", [exe_path, destination]),
        ]
        if manual_flag:
            commands.append(
                (
                    "part2alt_manual",
                    [
                        os.path.join(
                            executable_root,
                            "Tenoke_ccta_Part2Alt",
                            "Tenoke_ccta_Part2Alt.exe",
                        ),
                        destination,
                    ],
                )
            )
        commands.extend(
            [
                (
                    "part2altb",
                    [
                        os.path.join(
                            executable_root,
                            "Tenoke_ccta_Part2AltB",
                            "Tenoke_ccta_Part2AltB.exe",
                        ),
                        destination,
                    ],
                ),
                (
                    "part3",
                    [
                        os.path.join(
                            executable_root,
                            "Tenoke_ccta_Part3",
                            "Tenoke_ccta_Part3.exe",
                        ),
                        destination,
                    ],
                ),
                (
                    "part4",
                    [
                        os.path.join(
                            executable_root,
                            "Tenoke_ccta_Part4",
                            "Tenoke_ccta_Part4.exe",
                        ),
                        destination,
                        "45",
                    ],
                ),
            ]
        )
        return commands

    @staticmethod
    def _terminate_process_tree(process, wait_seconds=5):
        """Terminate descendants before their parent and return cleanup errors."""

        cleanup_errors = []
        if process.poll() is not None:
            return cleanup_errors

        if psutil is None:
            try:
                process.kill()
            except Exception as error:
                cleanup_errors.append(str(error))
            return cleanup_errors

        try:
            parent = psutil.Process(process.pid)
            descendants = parent.children(recursive=True)
            processes = list(reversed(descendants))
            processes.append(parent)

            for child_process in processes:
                try:
                    child_process.terminate()
                except psutil.NoSuchProcess:
                    pass
                except Exception as error:
                    cleanup_errors.append(str(error))

            _, alive = psutil.wait_procs(processes, timeout=wait_seconds)
            for child_process in alive:
                try:
                    child_process.kill()
                except psutil.NoSuchProcess:
                    pass
                except Exception as error:
                    cleanup_errors.append(str(error))
            if alive:
                psutil.wait_procs(alive, timeout=wait_seconds)
        except psutil.NoSuchProcess:
            pass
        except Exception as error:
            cleanup_errors.append(str(error))
            try:
                process.kill()
            except Exception as kill_error:
                cleanup_errors.append(str(kill_error))

        return cleanup_errors

    @staticmethod
    def _write_live_output(text):
        """Write vendor output immediately without letting console errors stop a case."""

        if not text:
            return
        output_stream = sys.stdout
        try:
            output_stream.write(text)
        except UnicodeEncodeError:
            encoding = getattr(output_stream, "encoding", None) or "utf-8"
            safe_text = text.encode(encoding, errors="replace").decode(
                encoding, errors="replace"
            )
            output_stream.write(safe_text)
        except Exception:
            return
        try:
            output_stream.flush()
        except Exception:
            pass

    @classmethod
    def _run_stage_safely(cls, stage_name, arguments, timeout_seconds, output_limit):
        """Run one vendor stage with live output and a bounded diagnostic tail."""

        result = {
            "name": stage_name,
            "executable": arguments[0],
            "arguments": list(arguments),
            "returncode": None,
            "timed_out": False,
            "duration_seconds": 0.0,
            "output": "",
            "error": "",
        }
        process = None
        output_thread = None
        output_tail = bytearray()
        output_tail_lock = threading.Lock()
        output_reader_errors = []
        started_at = time.monotonic()
        output_limit = max(0, int(output_limit))

        command_text = subprocess.list2cmdline([str(value) for value in arguments])
        cls._write_live_output(
            "[Centerline] starting {0}: {1}\n".format(stage_name, command_text)
        )

        def stream_output():
            encoding = locale.getpreferredencoding(False) or "utf-8"
            decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
            try:
                while True:
                    output_chunk = process.stdout.read(4096)
                    if not output_chunk:
                        break
                    if output_limit:
                        with output_tail_lock:
                            output_tail.extend(output_chunk)
                            excess_length = len(output_tail) - output_limit
                            if excess_length > 0:
                                del output_tail[:excess_length]
                    cls._write_live_output(decoder.decode(output_chunk))
                cls._write_live_output(decoder.decode(b"", final=True))
            except Exception as error:
                output_reader_errors.append(str(error))
            finally:
                try:
                    process.stdout.close()
                except Exception:
                    pass

        try:
            creation_flags = 0
            if os.name == "nt":
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

            try:
                process = subprocess.Popen(
                    list(arguments),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    creationflags=creation_flags,
                    bufsize=0,
                )
                output_thread = threading.Thread(
                    target=stream_output,
                    name="centerline-output-{0}".format(stage_name),
                )
                output_thread.daemon = True
                output_thread.start()
                try:
                    process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    result["timed_out"] = True
                    cleanup_errors = cls._terminate_process_tree(process)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            process.kill()
                        except Exception as error:
                            cleanup_errors.append(str(error))
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            cleanup_errors.append(
                                "process remained alive after forced termination"
                            )
                    result["error"] = "Timed out after {0} seconds".format(
                        timeout_seconds
                    )
                    if cleanup_errors:
                        result["error"] += "; cleanup: {0}".format(
                            "; ".join(cleanup_errors)
                        )
                result["returncode"] = process.poll()
            except (OSError, ValueError) as error:
                result["error"] = "Could not start process: {0}".format(error)
            except Exception as error:
                if process is not None and process.poll() is None:
                    cleanup_errors = cls._terminate_process_tree(process)
                    try:
                        process.wait(timeout=10)
                    except Exception as wait_error:
                        cleanup_errors.append(str(wait_error))
                result["error"] = "Process error: {0}".format(error)
        except Exception as error:
            if process is not None and process.poll() is None:
                cls._terminate_process_tree(process)
            result["error"] = result["error"] or "Log capture error: {0}".format(error)

        if output_thread is not None:
            output_thread.join(timeout=10)
            if output_thread.is_alive() and process.stdout is not None:
                try:
                    process.stdout.close()
                except Exception:
                    pass
                output_thread.join(timeout=1)
            if output_thread.is_alive():
                output_reader_errors.append("output reader did not stop")

        with output_tail_lock:
            captured_output = bytes(output_tail)
        encoding = locale.getpreferredencoding(False) or "utf-8"
        result["output"] = captured_output.decode(
            encoding, errors="replace"
        ).strip()
        if output_reader_errors:
            cls._write_live_output(
                "[Centerline] output capture warning ({0}): {1}\n".format(
                    stage_name, "; ".join(output_reader_errors)
                )
            )

        result["duration_seconds"] = round(time.monotonic() - started_at, 3)
        if not result["error"] and result["returncode"] != 0:
            result["error"] = "Exited with return code {0}".format(result["returncode"])
        if result["error"]:
            completion_text = "failed: {0}".format(result["error"])
        else:
            completion_text = "completed with return code {0}".format(
                result["returncode"]
            )
        cls._write_live_output(
            "[Centerline] {0} {1} ({2:.3f} seconds)\n".format(
                stage_name,
                completion_text,
                result["duration_seconds"],
            )
        )
        return result

    def run_case_safely(
        self,
        destination=None,
        exe_path=None,
        timeout_seconds=3600,
        output_limit=4000,
    ):
        """Run one complete vendor pipeline without aborting the caller's batch.

        The returned dictionary contains the overall ``success`` / ``exe_failed``
        status, one bounded diagnostic string, and a result for every attempted
        EXE stage.  A failed stage stops only this case; no ordinary launch,
        timeout, or non-zero-return failure is raised to the caller.
        """

        destination = self.destination if destination is None else destination
        exe_path = self.part1_address if exe_path is None else exe_path
        result = {
            "destination": "" if destination is None else str(destination),
            "manual": False,
            "success": False,
            "exe_failed": True,
            "error_message": "",
            "stages": [],
        }

        try:
            timeout_seconds = float(timeout_seconds)
            if timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be greater than zero")
            output_limit = int(output_limit)
            if output_limit < 0:
                raise ValueError("output_limit must not be negative")
            if not str(exe_path or "").strip():
                raise ValueError("Part 1 executable path is required")
            if not str(destination or "").strip():
                raise ValueError("Case destination is required")

            destination = os.path.abspath(str(destination))
            manual_flag = os.path.isfile(
                os.path.join(destination, "Combo_Vals_Man.nii.gz")
            )
            result["destination"] = destination
            result["manual"] = manual_flag

            commands = self._safe_stage_commands(exe_path, destination, manual_flag)
            for stage_name, arguments in commands:
                stage_result = self._run_stage_safely(
                    stage_name,
                    arguments,
                    timeout_seconds,
                    output_limit,
                )
                result["stages"].append(stage_result)
                if stage_result["error"]:
                    message = "{0}: {1}".format(stage_name, stage_result["error"])
                    if stage_result["output"]:
                        message += "; output: {0}".format(stage_result["output"])
                    result["error_message"] = message
                    return result

            result["success"] = True
            result["exe_failed"] = False
            return result
        except Exception as error:
            result["error_message"] = "Runner error: {0}".format(error)
            return result

    # generate the windows command to generate the center line
    def run(self):
        self.DetectManual()  # detect if manual tracing file exists
        self.rearrange()  # rearrange command info
        step1 = self.part1_address + ' ' + self.destination
        step105 = self.part105_address + ' ' + self.destination
        step2 = self.part2_address + ' ' + self.destination
        step3 = self.part3_address + ' ' + self.destination
        step4 = self.part4_address + ' ' + self.destination + ' 45'

        # print(step1)
        # print('\n')
        #
        # if self.ManualFlag:
        #     print(step105)
        #     print('\n')
        # else:
        #     pass

        # print(step2)
        # print('\n')
        # print(step3)
        # print('\n')
        # print(step4)
        print(step1)
        os.system(step1)
        print('step 1 done')

        # print('Manual file {}'.format(self.ManualFlag))

        if self.ManualFlag:
            os.system(step105)
            print('manual tracing done')
        else:
            pass

        print(step2)
        os.system(step2)
        print('step 2 done')
        print(step3)
        os.system(step3)
        print('step 3 done')
        print(step4)
        os.system(step4)
        print('all steps done')
        # time.sleep(5)


# create multi thread to allow process multiple cases at the same time, usually two threads are maximum
class myThread(threading.Thread):
    def __init__(self, threadID=0, name='', folderPath='', exePath='', jumpFlag=False):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.folderPath = folderPath
        self.exePath = exePath
        self.jumpFlag = jumpFlag

    def run(self):
        count = 0
        print('starting' + self.name)
        for i in range(len(self.folderPath)):
            a = CenterlineEx(exePath=self.exePath, folderPath=self.folderPath[i], jumpFlag=self.jumpFlag)
            a.run()
            print(str(self.threadID) + self.folderPath[i])
            # time.sleep(15)
        print('Exiting' + self.name)
