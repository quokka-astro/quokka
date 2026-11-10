import argparse
import re
import os
import shlex
import subprocess
import sys
import email
import smtplib

USAGE = """
The test suite and its tests are defined through an input file in an INI
configuration file format.

The "main" block specifies the global test suite parameters:

  [main]

  verbose        = < repeat captured errors to stderr, e.g., for CI runs >

  testTopDir     = < full path to test output directory >
  webTopDir      = < full path to test web output directory >
  wallclockFile  = < name of json file for storing past runtimes, to which .json will be appended;
                     set to wallclock_history by default >

  useCmake       = < 0: GNU Make handles the build (default)
                     1: CMake handles the build >

  isSuperbuild   = < 0: pre-build AMReX and source (default)
                     1: CMake downloads AMReX and needs separate configure & build >

  updateGitSubmodules = < 0: don't update submodules when changing git branches (default)
                       1: run `git submodule update --init` after changing git branches >

  sourceTree = < C_Src or AMReX >

  suiteName = < descriptive name (i.e. Castro) >

  reportActiveTestsOnly = <0: web shows every test ever run;
                           1: just current tests >

  goUpLink = <1: add "Go UP" link at top of the web page >

  COMP  = < name of C/C++ compiler, e.g., g++ >

  add_to_c_make_command = < any additional defines to add to the make invocation for C_Src AMReX >

  archive_output = <0: leave all plotfiles in place;
                    1: tar plotfiles after compare (default) >
  purge_output = <0: leave all plotfiles in place;
                  1: delete plotfiles after compare >

  MAKE = < name of make >
  numMakeJobs = < number of make jobs >

  MPIcommand = < MPI run command, with holders for host, # of proc, command >

     This should look something like:

          mpiexec -host @host@ -n @nprocs@ @command@ >

  MPIhost = < host for MPI job -- depends on MPI implementation >

  sendEmailWhenFail = < 1: send email when any tests fail >

  emailTo = < list of email addresses separated by commas, such as,
              foo@example.com, bar@example.com >

  emailBody = < email body >


The source git repositories are defined in separate blocks.  There
will always be a "AMReX" block, and usually a "source" block which is
the default directory used for compiling the tests.  Any extra repos
(including those where additional tests are to be build) are defined
in their own block starting with "extra-"

The general form is:

  [name]

  dir = < full path to git repo >

  branch = < desired branch in the git repo >

  build = < 1: this is a directory that tests will be compiled in >

  cmakeSetupOpts = < Options for CMake Setup (used only if useCmake=1) >

  cmake = < cmake command >

      This can be changed to specify a specific CMake verison. i.e. if
      you have ver. 3.17 installed as `cmake_3.17`, write
      cmake = cmake_3.17

  comp_string = < a string that is added to the make line >

      comp_string can refer to both the main source directory (as @source@)
      and its own directory (as @self@), for example:

      comp_string = CASTRO_DIR=@source@ WDMERGER_HOME=@self@


Each test is given its own block, with the general form:

  [Sod-x]

  buildDir = < relative path (from sourceDir) for this problem >

  target = <name of the make target associated to the test (cmake only)>

  inputFile = < input file name >
  probinFile = < probin file name >

  dim = < dimensionality: 1, 2, or 3 >

  aux?File = < name of additional file needed by the test >
  link?File = < name of additional file needed by the test >

      Here "?" is 1, 2, or 3, allowing for several files per test

  restartTest = < is this a restart test? 0 for no, 1 for yes >
  restartFileNum = < # of file to restart from (if restart test) >

  useMPI = <is this a parallel (MPI) job? 0 for no, 1 for yes) >
  numprocs = < # of processors to run on (if parallel job) >

  useOMP = <is this an OpenMP job? 0 for no, 1 for yes) >
  numthreads = < # of threads to us with OpenMP (if OpenMP job) >

  acc = < 0 for normal run, 1 if we want OpenACC >

  debug = < 0 for normal run, 1 if we want debugging options on >

  compileTest = < 0 for normal run, 1 if we just test compilation >

  selfTest = < 0 for normal run, 1 if test self-diagnoses if it succeeded >
  stSuccessString = < string to find in self-test output to determine success >

  doVis = < 0 for no visualization, 1 if we do visualization >
  visVar = < string of the variable to visualize >

  analysisRoutine = < name of the script to run on the output >

      The script is run as:

        analysisRoutine [options] plotfile

  analysisMainArgs = < commandline arguments to pass to the analysisRoutine --
                       these should refer to options from the [main] block >

  analysisOutputImage = < name on analysis result image to show on web page >

  compareFile = < explicit output file to do the comparison with -- this is
                  assumed to be prefixed with the test name when output by
                  the code at runtime, e.g. test_plt00100 >

  doComparison = < 1: compare to benchmark file, 0: skip comparison >
  tolerance = < floating point number representing the largest relative error
                permitted between the run output and the benchmark for mesh data,
                default is 0.0 >
  abs_tolerance = < floating point number representing the largest absolute
                    error permitted between the run output and the benchmark for
                    mesh data, default is 0.0 >
  particle_tolerance = < same as tolerance, for particle comparisons >
  particle_abs_tolerance = < same as tolerance, for particle comparisons >
  outputFile = < explicit output file to compare with -- exactly as it will
                 be written.  No prefix of the test name will be done >

  diffDir = < directory/file to do a plain text diff on (recursive, if dir) >

  diffOpts = < options to use with the diff command for the diffDir comparison >

  check_performance = < 1: compare run time of test to average of past runs >
  performance_threshold = < ratio of run time / running average above which a
                            a performance warning will be issued, default is
                            1.2 >
  runs_to_average = < number of past runs to include when computing the average,
                      default is 5 >

  runtime_params =  < run-time parameters to be added to the test command line
                      arguments >

Getting started:

To set up a test suite, it is probably easiest to write the
testfile.ini as described above and then run the test routine with the
--make_benchmarks option to create the benchmark directory.
Subsequent runs can be done as usual, and will compare to the newly
created benchmarks.  If differences arise in the comparisons due to
(desired) code changes, the benchmarks can be updated using
--make_benchmarks to reflect the new ``correct'' solution.

"""


class Log:
    """a simple logging class to show information to the terminal"""
    def __init__(self, output_file=None):

        # http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
        # which in-turn cites the blender build scripts
        self.warn_color = '\033[33m'
        self.success_color = '\033[32m'
        self.fail_color = '\033[31m'
        self.bold_color = '\033[1m'
        self.end_color = '\033[0m'

        self.current_indent = 0
        self.indent_str = ""

        if output_file is not None:
            try:
                self.of = open(output_file, "w")
            except OSError:
                print("ERROR: unable to open output file")
                raise OSError
            else:
                self.have_log = True
        else:
            self.of = None
            self.have_log = False

        self.suite = None

    def indent(self):
        """indent the log output by one stop"""
        self.current_indent += 1
        self.indent_str = self.current_indent*"   "

    def flush(self):
        """ flush the output file (if it exists) """
        if self.have_log:
            self.of.flush()

    def outdent(self):
        """undo one level of indent"""
        self.current_indent -= 1
        self.current_indent = max(0, self.current_indent)
        self.indent_str = self.current_indent*"   "

    def fail(self, string):
        """output a failure message to the log"""
        nstr = self.fail_color + string + self.end_color
        print(f"{self.indent_str}{nstr}")
        if self.have_log:
            self.of.write(f"{self.indent_str}{string}\n")

        def email_developers():
            pattern = r'(\/.*?\.out)'
            filename = re.findall(pattern, string)
            error_file = open(filename[0]) if filename else None
            error_string = error_file.read().encode("utf-8") if error_file else ""

            emailto = ",".join(self.suite.emailTo)
            emailfrom = self.suite.emailFrom
            subject = "Regression testing suite failed to run for {}".format(
                self.suite.suiteName)
            msg = "To: {} \nFrom: {} \nSubject: {}\n\n Reason: {}\n\n{}".format(
                emailto, emailfrom, subject, string, error_string)

            server = smtplib.SMTP('localhost')
            server.sendmail(self.suite.emailFrom, self.suite.emailTo, msg)
            server.quit()

        if self.suite.sendEmailWhenFail:
            self.suite.log.skip()
            self.suite.log.bold("sending email...")
            email_developers()

        self.close_log()
        sys.exit(1)

    def testfail(self, string):
        """output a test failure to the log"""
        nstr = self.fail_color + string + self.end_color
        print(f"{self.indent_str}{nstr}")
        if self.have_log:
            self.of.write(f"{self.indent_str}{string}\n")

    def warn(self, warn_msg):
        """
        output a warning.  It is always prefix with 'WARNING:'
        For multi-line warnings, send in a list of strings
        """
        prefix = self.indent_str + "WARNING: "
        filler = self.indent_str + "         "
        if isinstance(warn_msg, list):
            msg = [prefix + warn_msg[0]] + [filler + x for x in warn_msg[1:]]
            omsg = "\n".join(msg).strip()
        else:
            omsg = prefix + warn_msg
        nstr = self.warn_color + omsg + self.end_color
        print(nstr)
        if self.have_log:
            self.of.write(f"{omsg}\n")

    def success(self, string):
        """output a success message to the log"""
        nstr = self.success_color + string + self.end_color
        print(f"{self.indent_str}{nstr}")
        if self.have_log:
            self.of.write(f"{self.indent_str}{string}\n")

    def log(self, string):
        """output some text to the log"""
        print(f"{self.indent_str}{string}")
        if self.have_log:
            self.of.write(f"{self.indent_str}{string}\n")

    def skip(self):
        """introduce a newline in the log"""
        print("")
        if self.have_log:
            self.of.write("\n")

    def bold(self, string):
        """emphasize a log message"""
        nstr = self.bold_color + string + self.end_color
        print(f"{self.indent_str}{nstr}")
        if self.have_log:
            self.of.write(f"{self.indent_str}{string}\n")

    def close_log(self):
        """close the log"""
        if self.have_log:
            self.of.close()


def get_args(arg_string=None):
    """ parse the commandline arguments.  If arg_string is present, we
        parse from there, otherwise we use the default (sys.argv) """

    parser = argparse.ArgumentParser(description=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    tests_group = parser.add_argument_group("test selection",
                                            "options that determine which tests to run")
    tests_group.add_argument("--single_test", type=str, default="", metavar="test-name",
                             help="name of a single test to run")
    tests_group.add_argument("--tests", type=str, default="", metavar="'test1 test2 test3'",
                             help="a space-separated list of tests to run")
    tests_group.add_argument("-d", type=int, default=-1,
                             help="restrict tests to a particular dimensionality")
    tests_group.add_argument("--redo_failed", action="store_true",
                             help="only run the tests that failed last time")
    tests_group.add_argument("--keyword", type=str, default=None,
                             help="run tests only with this keyword specified in their definitions")

    git_group = parser.add_argument_group("git options",
                                          "options that control how we interact the git repos")
    git_group.add_argument("--no_update", type=str, default="None", metavar="name",
                           help="which codes to exclude from the git update?" +
                           " (None, All, or a comma-separated list of codes)")
    git_group.add_argument("--source_branch", type=str, default=None, metavar="branch-name",
                           help="what git branch to use for the source repo")
    git_group.add_argument("--source_pr", type=int, default=None, metavar="PR-number",
                           help="what github pull request number to use for the source repo")
    git_group.add_argument("--amrex_pr", type=int, default=None, metavar="PR-number",
                           help="what github pull request number to use for the amrex repo")

    bench_group = parser.add_argument_group("benchmark options",
                                            "options that control benchmark creation")
    bench_group.add_argument("--make_benchmarks", type=str, default=None, metavar="comment",
                             help="make new benchmarks? (must provide a comment)")
    bench_group.add_argument("--copy_benchmarks", type=str, default=None, metavar="comment",
                             help="use plotfiles from failed tests of the last run as new benchmarks." +
                             " No git pull is done and no new runs are performed (must provide a comment)")

    run_group = parser.add_argument_group("test running options",
                                          "options that control how the tests are run")
    run_group.add_argument("--compile_only", action="store_true",
                           help="test only that the code compiles, without running anything")
    run_group.add_argument("--with_valgrind", action="store_true",
                           help="run with valgrind")
    run_group.add_argument("--valgrind_options", type=str, default="--leak-check=yes --log-file=vallog.%p",
                           help="valgrind options", metavar="'valgrind options'")

    suite_options = parser.add_argument_group("suite options",
                                              "options that control the test suite operation")
    suite_options.add_argument("--do_temp_run", action="store_true",
                               help="is this a temporary run? (output not stored or logged)")
    suite_options.add_argument("--send_no_email", action="store_true",
                               help="do not send emails when tests fail")
    suite_options.add_argument("--with_coverage", action="store_true",
                               help="report parameter coverage for this test run")
    suite_options.add_argument("--check_performance", nargs=2,
                               metavar=("performance_threshold", "runs_to_average"),
                               help="measure the performance of each test run against the last runs_to_average runs, "
                               + "supplying a warning on a ratio greater than performance_threshold")
    suite_options.add_argument("--note", type=str, default="",
                               help="a note on the resulting test webpages")
    suite_options.add_argument("--complete_report_from_crash", type=str, default="", metavar="testdir",
                               help="complete report generation from a crashed test suite run named testdir")
    suite_options.add_argument("--log_file", type=str, default=None, metavar="logfile",
                               help="log file to write output to (in addition to stdout")
    suite_options.add_argument("--clean_testdir", action="store_true",
                               help="remove individual test directory after each passed test")
    suite_options.add_argument("--delete_exe", action="store_true",
                               help="remove executable in test directory after each passed test")

    comp_options = parser.add_argument_group("comparison options",
                                             "options that control how the comparisons are done")
    comp_options.add_argument("--skip_comparison", action="store_true",
                              help="run analysis for each test without comparison to benchmarks")
    comp_options.add_argument("--tolerance", type=float, default=None, metavar="value",
                              help="largest relative error permitted during mesh comparison")
    comp_options.add_argument("--abs_tolerance", type=float, default=None, metavar="value",
                              help="largest absolute error permitted during mesh comparison")
    comp_options.add_argument("--particle_tolerance", type=float, default=None, metavar="value",
                              help="largest relative error permitted during particle comparison")
    comp_options.add_argument("--particle_abs_tolerance", type=float, default=None, metavar="value",
                              help="largest absolute error permitted during particle comparison")
    parser.add_argument("input_file", metavar="input-file", type=str, nargs=1,
                        help="the input file (INI format) containing the suite and test parameters")

    if not arg_string is None:
        args = parser.parse_args(arg_string)
    else:
        args = parser.parse_args()

    return args


def run(string, stdin=False, outfile=None, store_command=False, env=None,
        outfile_mode="a", errfile=None, log=None, cwd=None):

    # shlex.split will preserve inner quotes
    prog = shlex.split(string)
    sin = None
    if stdin: sin = subprocess.PIPE

    p0 = subprocess.Popen(prog, stdin=sin, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=env, cwd=cwd)

    stdout0, stderr0 = p0.communicate()
    if stdin: p0.stdin.close()
    rc = p0.returncode
    p0.stdout.close()
    p0.stderr.close()

    stdout0 = stdout0.decode('utf-8')
    stderr0 = stderr0.decode('utf-8')

    if outfile is not None:
        try: cf = open(outfile, outfile_mode)
        except OSError:
            log.fail("  ERROR: unable to open file for writing")
        else:
            if store_command:
                cf.write(string)
                cf.write('\n')
            for line in stdout0:
                cf.write(line)

            if errfile is None:
                for line in stderr0:
                    cf.write(line)

            cf.close()

    if errfile is not None and stderr0 is not None:
        write_err = True
        if isinstance(stderr0, str):
            if stderr0.strip() == "":
                write_err = False
        if write_err:
            try: cf = open(errfile, outfile_mode)
            except OSError:
                log.fail("  ERROR: unable to open file for writing")
            else:
                for line in stderr0:
                    cf.write(line)
                cf.close()

    return stdout0, stderr0, rc


def get_recent_filename(fdir, base, extension):
    """ find the most recent file matching the base and extension """

    files = [f for f in os.listdir(fdir) if (f.startswith(base) and
                                             f.endswith(extension))]

    files.sort(key=lambda x: os.path.getmtime(os.path.join(fdir, x)))

    try:
        return files.pop()
    except:
        return None
