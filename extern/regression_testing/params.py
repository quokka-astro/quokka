try: import ConfigParser as configparser
except ImportError:
    import configparser   # python 3

import getpass
import os
import socket
import re

import repo
import suite
import test_util

def convert_type(string):
    """ return an integer, float, or string from the input string """
    if string is None:
        return None

    try: int(string)
    except: pass
    else: return int(string)

    try: float(string)
    except: pass
    else: return float(string)

    return string.strip()

def safe_get(cp, sec, opt, default=None):
    try: v = cp.get(sec, opt)
    except: v = default
    return v

def load_params(args):
    """
    reads the parameter file and creates as list of test objects as well as
    the suite object
    """

    test_list = []

    try:
        cp = configparser.ConfigParser(strict=False)
    except:
        cp = configparser.ConfigParser()

    cp.optionxform = str

    log = test_util.Log(output_file=args.log_file)

    log.bold("loading " + args.input_file[0])

    if not os.path.exists(args.input_file[0]):
        raise OSError(f"Parameter file {args.input_file[0]} does not exist")
    
    try: cp.read(args.input_file[0])
    except:
        log.fail(f"ERROR: unable to read parameter file {args.input_file[0]}")

    # "main" is a special section containing the global suite parameters.
    mysuite = suite.Suite(args)
    log.suite = mysuite
    mysuite.log = log

    valid_options = list(mysuite.__dict__.keys())

    for opt in cp.options("main"):

        # get the value of the current option
        value = convert_type(cp.get("main", opt))

        if opt in valid_options or "_" + opt in valid_options:

            if opt == "sourceTree":
                if not value in ["C_Src", "AMReX", "amrex"]:
                    mysuite.log.fail("ERROR: invalid sourceTree")
                else:
                    mysuite.sourceTree = value

            elif opt == "testTopDir":
                mysuite.testTopDir = mysuite.check_test_dir(value)
                print(f"just set testTopDir = {mysuite.testTopDir}")
            elif opt == "webTopDir":
                mysuite.init_web_dir(value)
            elif opt == "reportCoverage":
                mysuite.reportCoverage = mysuite.reportCoverage or value
            elif opt == "emailTo":
                mysuite.emailTo = value.split(",")
            elif opt == "ftools":
                mysuite.ftools = value.split()
            elif opt == "extra_tools":
                mysuite.extra_tools = value

            else:
                # generic setting of the object attribute
                setattr(mysuite, opt, value)

        else:
            mysuite.log.warn(f"suite parameter {opt} not valid")


    # AMReX -- this will always be defined
    rdir = mysuite.check_test_dir(safe_get(cp, "AMReX", "dir"))

    branch = safe_get(cp, "AMReX", "branch")  # note: always as string
    rhash = convert_type(safe_get(cp, "AMReX", "hash"))

    mysuite.repos["AMReX"] = repo.Repo(mysuite, rdir, "AMReX",
                                       branch_wanted=branch, hash_wanted=rhash)

    if args.amrex_pr is not None:
        mysuite.repos["AMReX"].pr_wanted = args.amrex_pr

    # Check for Cmake build options for both AMReX and Source
    for s in cp.sections():
        if s == "AMReX":
            mysuite.amrex_cmake_opts = safe_get(cp, s, "cmakeSetupOpts", default= "")
        elif s == "source":
            mysuite.source_cmake_opts = safe_get(cp, s, "cmakeSetupOpts", default= "")

    # now all the other build and source directories
    other_srcs = [s for s in cp.sections() if s.startswith("extra-")]
    if not mysuite.sourceTree in ["AMReX", "amrex"]: other_srcs.append("source")

    for s in other_srcs:
        if s.startswith("extra-"):
            k = s.split("-",1)[1]
        else:
            k = "source"

        rdir = mysuite.check_test_dir(safe_get(cp, s, "dir"))
        branch = safe_get(cp, s, "branch")  # note: always as string
        rhash = convert_type(safe_get(cp, s, "hash"))


        build = convert_type(safe_get(cp, s, "build", default=0))
        if s == "source": build = 1

        comp_string = safe_get(cp, s, "comp_string")

        name = os.path.basename(os.path.normpath(rdir))

        mysuite.repos[k] = repo.Repo(mysuite, rdir, name,
                                     branch_wanted=branch, hash_wanted=rhash,
                                     build=build, comp_string=comp_string)


    # AMReX-only tests don't have a sourceDir
    mysuite.amrex_dir = mysuite.repos["AMReX"].dir

    if mysuite.sourceTree in ["AMReX", "amrex"]:
        mysuite.source_dir = mysuite.repos["AMReX"].dir
    else:
        mysuite.source_dir = mysuite.repos["source"].dir

    # did we override the branch on the commandline?
    if args.source_branch is not None and args.source_pr is not None:
        mysuite.log.fail("ERROR: cannot specify both source_branch and source_pr")

    if args.source_branch is not None:
        mysuite.repos["source"].branch_wanted = args.source_branch

    if args.source_pr is not None:
        mysuite.repos["source"].pr_wanted = args.source_pr

    # now flesh out the compile strings -- they may refer to either themselves
    # or the source dir
    for r in mysuite.repos.keys():
        s = mysuite.repos[r].comp_string
        if not s is None:
            mysuite.repos[r].comp_string = \
                s.replace("@self@", mysuite.repos[r].dir).replace("@source@", mysuite.repos["source"].dir)

    # the suite needs to know any ext_src_comp_string
    for r in mysuite.repos.keys():
        if not mysuite.repos[r].build == 1:
            if not mysuite.repos[r].comp_string is None:
                mysuite.extra_src_comp_string += f" {mysuite.repos[r].comp_string} "

    # checks
    if args.send_no_email:
        mysuite.sendEmailWhenFail = 0

    if mysuite.sendEmailWhenFail:
        if mysuite.emailTo == [] or mysuite.emailBody == "":
            mysuite.log.fail("ERROR: when sendEmailWhenFail = 1, you must specify emailTo and emailBody\n")

        if mysuite.emailFrom == "":
            mysuite.emailFrom = '@'.join((getpass.getuser(), socket.getfqdn()))

        if mysuite.emailSubject == "":
            mysuite.emailSubject = mysuite.suiteName+" Regression Test Failed"

    if mysuite.slack_post:
        if not os.path.isfile(mysuite.slack_webhookfile):
            mysuite.log.warn("slack_webhookfile invalid")
            mysuite.slack_post = 0
        else:
            print(mysuite.slack_webhookfile)
            try: f = open(mysuite.slack_webhookfile)
            except:
                mysuite.log.warn("unable to open webhook file")
                mysuite.slack_post = 0
            else:
                mysuite.slack_webhook_url = str(f.readline())
                f.close()

    if (mysuite.sourceTree == "" or mysuite.amrex_dir == "" or
        mysuite.source_dir == "" or mysuite.testTopDir == ""):
        mysuite.log.fail("ERROR: required suite-wide directory not specified\n" + \
                         "(sourceTree, amrexDir, sourceDir, testTopDir)")

    # Make sure the web dir is valid
    if not os.path.isdir(mysuite.webTopDir):
        try: os.mkdir(mysuite.webTopDir)
        except:
            mysuite.log.fail("ERROR: unable to create the web directory: {}\n".format(
                mysuite.webTopDir))

    # all other sections are tests
    mysuite.log.skip()
    mysuite.log.bold("finding tests and checking parameters...")

    for sec in cp.sections():

        if sec in ["main", "AMReX", "source"] or sec.startswith("extra-"): continue

        # maximum test name length -- used for HTML formatting
        mysuite.lenTestName = max(mysuite.lenTestName, len(sec))

        # create the test object for this test
        mytest = suite.Test(sec)
        mytest.log = log
        invalid = 0

        # set the test object data by looking at all the options in
        # the current section of the parameter file
        valid_options = list(mytest.__dict__.keys())
        aux_pat = re.compile(r"aux\d+File", re.IGNORECASE)
        link_pat = re.compile(r"link\d+File", re.IGNORECASE)

        for opt in cp.options(sec):

            # get the value of the current option
            value = convert_type(cp.get(sec, opt))

            if opt in valid_options or "_" + opt in valid_options:

                if opt == "keyword":
                    mytest.keywords = [k.strip() for k in value.split(",")]

                else:
                    # generic setting of the object attribute
                    setattr(mytest, opt, value)
            
            elif aux_pat.match(opt):
                
                mytest.auxFiles.append(value)
                
            elif link_pat.match(opt):
                
                mytest.linkFiles.append(value)

            else:
                
                mysuite.log.warn(f"unrecognized parameter {opt} for test {sec}")


        # make sure that the build directory actually exists
        if not mytest.extra_build_dir == "":
            bdir = mysuite.repos[mytest.extra_build_dir].dir + mytest.buildDir
        else:
            bdir = mysuite.source_dir + mytest.buildDir

        if not os.path.isdir(bdir):
            mysuite.log.warn(f"invalid build directory: {bdir}")
            invalid = 1


        # make sure all the require parameters are present
        if mytest.compileTest:
            if mytest.buildDir == "":
                mysuite.log.warn(f"mandatory parameters for test {sec} not set")
                invalid = 1

        else:
            
            input_file_invalid = mytest.inputFile == "" and not mytest.run_as_script
            if mytest.buildDir == "" or input_file_invalid or mytest.dim == -1:
                warn_msg = [f"required params for test {sec} not set",
                            f"buildDir = {mytest.buildDir}",
                            f"inputFile = {mytest.inputFile}"]
                warn_msg += [f"dim = {mytest.dim}"]
                mysuite.log.warn(warn_msg)

                invalid = 1

        # check the optional parameters
        if mytest.restartTest and mytest.restartFileNum == -1:
            mysuite.log.warn(f"restart-test {sec} needs a restartFileNum")
            invalid = 1

        if mytest.selfTest and mytest.stSuccessString == "":
            mysuite.log.warn(f"self-test {sec} needs a stSuccessString")
            invalid = 1

        if mytest.useMPI and mytest.numprocs == -1:
            mysuite.log.warn(f"MPI parallel test {sec} needs numprocs")
            invalid = 1

        if mytest.useOMP and mytest.numthreads == -1:
            mysuite.log.warn(f"OpenMP parallel test {sec} needs numthreads")
            invalid = 1

        if mytest.doVis and mytest.visVar == "":
            mysuite.log.warn(f"test {sec} has visualization, needs visVar")
            invalid = 1

        if mysuite.sourceTree in ["AMReX", "amrex"] and mytest.testSrcTree == "":
            mysuite.log.warn(f"testSrcTree not set for AMReX test {sec}")
            invalid = 1


        # add the current test object to the master list
        if not invalid:
            test_list.append(mytest)
        else:
            mysuite.log.warn(f"test {sec} will be skipped")


    # if any runs are parallel, make sure that the MPIcommand is defined
    any_MPI = any([t.useMPI for t in test_list])

    if any_MPI and mysuite.MPIcommand == "":
        mysuite.log.fail("ERROR: some tests are MPI parallel, but MPIcommand not defined")

    test_list.sort()

    return mysuite, test_list
