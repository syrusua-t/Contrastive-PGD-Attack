from colorama import Fore
from colorama import Style

def cprint(content):
    print(f"->{Fore.RED} {content} {Style.RESET_ALL}<-")