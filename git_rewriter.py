import os

files = {
    'hypoid_test.py': [
        (b"target_points = base_points - (E.flatten(order = 'F')) * base_normals",
         b"target_points = base_points + (E.flatten(order = 'F')) * base_normals"),
        (b"EO = np.sum((base_points - target_points) * base_normals, axis = 0)",
         b"EO = np.sum((target_points - base_points) * base_normals, axis = 0)"),
        (b"target_points_cvx = base_points_cvx - (E_cvx.flatten(order='F')) * base_normals_cvx",
         b"target_points_cvx = base_points_cvx + (E_cvx.flatten(order='F')) * base_normals_cvx"),
        (b"EO_cvx = np.sum((base_points_cvx - target_points_cvx) * base_normals_cvx, axis=0)",
         b"EO_cvx = np.sum((target_points_cvx - base_points_cvx) * base_normals_cvx, axis=0)")
    ],
    'run_contact_check.py': [
        (b"R(\xe9\xbd\xbf\xe5\xae\xbd)", b"R(\xe9\xbd\xbf\xe9\xab\x98)"),
        (b"Z(\xe9\xbd\xbf\xe9\xab\x98)", b"Z(\xe9\xbd\xbf\xe5\xae\xbd)"),
        (b"\xe9\xbd\xbf\xe5\xae\xbd\xe4\xbd\x8d\xe7\xbd\xae: {R_pct:.1f}% (0%=\xe5\xb0\x8f\xe7\xab\xaftoe, 100%=\xe5\xa4\xa7\xe7\xab\xafheel)",
         b"\xe9\xbd\xbf\xe9\xab\x98\xe4\xbd\x8d\xe7\xbd\xae: {R_pct:.1f}% (0%=\xe9\xbd\xbf\xe6\xa0\xb9, 100%=\xe9\xbd\xbf\xe9\xa1\xb6)"),
        (b"\xe9\xbd\xbf\xe9\xab\x98\xe4\xbd\x8d\xe7\xbd\xae: {Z_pct:.1f}% (0%=\xe9\xbd\xbf\xe6\xa0\xb9, 100%=\xe9\xbd\xbf\xe9\xa1\xb6)",
         b"\xe9\xbd\xbf\xe5\xae\xbd\xe4\xbd\x8d\xe7\xbd\xae: {Z_pct:.1f}% (0%=\xe5\xb0\x8f\xe7\xab\xaftoe, 100%=\xe5\xa4\xa7\xe7\xab\xafheel)"),
        (b"if Z_pct < 30: pp", b"if R_pct < 30: pp"),
        (b"elif Z_pct > 70: pp", b"elif R_pct > 70: pp"),
        (b"if R_pct < 30: pf", b"if Z_pct < 30: pf"),
        (b"elif R_pct > 70: pf", b"elif Z_pct > 70: pf")
    ],
    'hypoid_contact.py': [
        (b"R(\xe9\xbd\xbf\xe5\xae\xbd)", b"R(\xe9\xbd\xbf\xe9\xab\x98)"),
        (b"Z(\xe9\xbd\xbf\xe9\xab\x98)", b"Z(\xe9\xbd\xbf\xe5\xae\xbd)"),
        (b"\xe9\xbd\xbf\xe5\xae\xbd\xe4\xbd\x8d\xe7\xbd\xae: {R_pct:.1f}% (0%=\xe5\xb0\x8f\xe7\xab\xaftoe, 100%=\xe5\xa4\xa7\xe7\xab\xafheel)",
         b"\xe9\xbd\xbf\xe9\xab\x98\xe4\xbd\x8d\xe7\xbd\xae: {R_pct:.1f}% (0%=\xe9\xbd\xbf\xe6\xa0\xb9, 100%=\xe9\xbd\xbf\xe9\xa1\xb6)"),
        (b"\xe9\xbd\xbf\xe9\xab\x98\xe4\xbd\x8d\xe7\xbd\xae: {Z_pct:.1f}% (0%=\xe9\xbd\xbf\xe6\xa0\xb9, 100%=\xe9\xbd\xbf\xe9\xa1\xb6)",
         b"\xe9\xbd\xbf\xe5\xae\xbd\xe4\xbd\x8d\xe7\xbd\xae: {Z_pct:.1f}% (0%=\xe5\xb0\x8f\xe7\xab\xaftoe, 100%=\xe5\xa4\xa7\xe7\xab\xafheel)"),
        (b"if Z_pct < 30:\n                    pos_profile", b"if R_pct < 30:\n                    pos_profile"),
        (b"elif Z_pct > 70:\n                    pos_profile", b"elif R_pct > 70:\n                    pos_profile"),
        (b"if R_pct < 30:\n                    pos_face", b"if Z_pct < 30:\n                    pos_face"),
        (b"elif R_pct > 70:\n                    pos_face", b"elif Z_pct > 70:\n                    pos_face")
    ]
}

def run_replace(forward=True):
    for f, replacements in files.items():
        with open(f, 'rb') as file:
            content = file.read()
        for new, old in replacements:
            if forward:
                content = content.replace(new, old)
            else:
                content = content.replace(old, new)
        with open(f, 'wb') as file:
            file.write(content)
    print("Replaced!")

import sys
if sys.argv[1] == 'revert':
    run_replace(True)
else:
    run_replace(False)
