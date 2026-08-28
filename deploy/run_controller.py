# Description: Plain Python entrypoint for the controller core without ROS dependencies

import os

from controller_core import ControllerCore


def configure_process_priority():
    pid = os.getpid()
    print("PID: ", pid)
    os.system("renice -n -21 -p " + str(pid))
    os.system("echo -20 > /proc/" + str(pid) + "/autogroup")


def main():
    configure_process_priority()
    controller = ControllerCore()
    print("Controller core initialized.")
    return controller


if __name__ == "__main__":
    main()
