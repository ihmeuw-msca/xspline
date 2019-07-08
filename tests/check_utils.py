# test suite for utils
import os
import sys
# add current directory
sys.path.append('./')


def run_test(name):
    namespace = {}
    exec('import ' + name, namespace)
    exec('ok = ' + name + '.' + name + '()', namespace)
    ok = namespace['ok']
    if ok:
        print(name + ': OK')
    else:
        print(name + ': Error')
    return ok


fun_list = [
    'utils_indexList',
    'utils_option2List',
    'utils_outerFlatten'
]

error_count = 0

for name in fun_list:
    ok = run_test(name)
    if not ok:
        error_count += 1

if error_count > 0:
    print('check_utils: error_count =', error_count)
    sys.exit(1)
else:
    print('check_utils: OK')
    sys.exit(0)
