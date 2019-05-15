# test suite for bspline
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
    'bspline_splineS',
    'bspline_splineF',
    'bspline_splineDF',
    'bspline_splineIF',
    'bspline_designMat',
    'bspline_designDMat',
    'bspline_lastDMat',
    'bspline_integrate'
]

error_count = 0

for name in fun_list:
    ok = run_test(name)
    if not ok:
        error_count += 1

if error_count > 0:
    print('check_bspline: error_count =', error_count)
    sys.exit(1)
else:
    print('check_bspline: OK')
    sys.exit(0)
