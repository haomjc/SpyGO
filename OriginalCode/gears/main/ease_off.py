import numpy as np
import matplotlib.pyplot as plt

# Enable interactive plotting mode
plt.ion()

def crowning_RZE(ra, rff, rf, b, Ep_tip, Ep_root, Ef_toe, Ef_heel, order_prof, order_face):
    internal_gears = False
    if ra < rf:
        internal_gears = True
    
    rm = (ra + rff)/2  # mean radius: the crowning is applied with respect to the mean radius rather than the pitch radius
    zm = 0

    ku_tip = Ep_tip/(np.abs(ra - rm))**(order_prof)
    ku_root = Ep_root/(np.abs(ra - rm))**(order_prof)
    kvtoe = Ef_toe/(b/2)**(order_face)
    kvheel = Ef_heel/(b/2)**(order_face)
    ubar =  np.abs(ra - rff)/2
    orderFillet = 0.8

    def ease_off(z, R):
        u = R - rm
        v = z - zm
        face_crowning = kvtoe*np.abs(v)**order_face*(v<=0) + kvheel*np.abs(v)**order_face*(v>0)
        kf = (ku_root*np.abs(ubar)**order_prof + face_crowning)/np.abs(rff - rf)**orderFillet

        if internal_gears:
            return (ku_tip*np.abs(u)**order_prof + face_crowning)*(R <= rm) + (ku_root*np.abs(u)**order_prof + face_crowning)*np.logical_and(R > rm, R<rff) + ( kf*(np.abs(R - rf))**orderFillet )*np.logical_and( R >= rff, R <= rf)
        else:
            return (ku_tip*np.abs(u)**order_prof + face_crowning)*(R >= rm) + (ku_root*np.abs(u)**order_prof + face_crowning)*np.logical_and(R < rm, R>rff) + ( kf*(np.abs(R - rf))**orderFillet )*np.logical_and( R <= rff, R >= rf)
        
    return ease_off
