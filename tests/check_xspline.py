# test suite for xspline
import os
import sys
# add the src and test directory
sys.path.append('./')
sys.path.append('../xspline/')


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
    'xspline_splineF',
    'xspline_splineDF',
    'xspline_splineIF',
    'xspline_designMat',
    'xspline_designDMat',
    'xspline_designIMat',
    'xspline_lastDMat',
    'xspline_integrate'
]

error_count = 0

for name in fun_list:
    ok = run_test(name)
    if not ok:
        error_count += 1

if error_count > 0:
    print('check_xspline: error_count =', error_count)
    sys.exit(1)
else:
    print('check_xspline: OK')
    sys.exit(0)
