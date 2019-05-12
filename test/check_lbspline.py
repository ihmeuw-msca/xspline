# test suite for lbspline
import os
import sys
# add the src and test directory
sys.path.append('./')
sys.path.append('../bspline/')


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
    'lbspline_intgLinear',
    'lbspline_splineF',
    'lbspline_splineDF',
    'lbspline_splineIF',
    'lbspline_designMat',
    'lbspline_designDMat',
    'lbspline_designIMat',
    'lbspline_lastDMat'
]

error_count = 0

for name in fun_list:
    ok = run_test(name)
    if not ok:
        error_count += 1

if error_count > 0:
    print('check_lbspline: error_count =', error_count)
    sys.exit(1)
else:
    print('check_lbspline: OK')
    sys.exit(0)
