import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from mpl_toolkits.mplot3d import Axes3D



dx = [0, 50, 100, 150, 200, 250, 300]

targets1 = dx+dx+dx
ferror = np.array([
0.779464882060509,
-3.25427565146556,
-6.95932617768823,
-12.0073101889155,
-15.1676738778516,
-19.0699383286756,
-22.8125549315977,
0.507121738850601,
-3.48905422811085,
-7.31618956480101,
-12.0260918975868,
-14.9798520422873,
-19.1826323797845,
-22.8501192987106,
0.582250862482144,
-3.4420985317772,
-7.2598429191547,
-12.4111287974768,
-15.026807738621,
-19.098112316338,
-22.8031626526066


])

berror=np.array([
-0.638597717077931,
-4.66294711133727,
-8.47130045442508,
-13.772844940923,
-16.8768616043046,
-21.2205109303434,
-24.9912996213872,
-0.375645713134753,
-4.55025325018244,
-8.57460260645097,
-13.9043207007031,
-17.0459026809679,
-21.0045141573464,
-25.1321667103882,
-0.309907728769939,
-4.68172948484777,
-8.45251808091458,
-13.7352805738101,
-16.8486885664125,
-21.1735552340097,
-25.2542511409475


])

ferror2 = np.array([
0.622945827797751,
-3.39514280378453,
-7.17845288721465,
-12.1481769613263,
-15.0581112195866,
-19.1168943415993,
-22.8219456276383


])

berror2= np.array([
-0.441383719660874,
-4.63164328212249,
-8.49947371393021,
-13.8041487384787,
-16.9238176172283,
-21.1328601072331,
-25.125905824241


])

targets2 = dx

error3 = np.array([
0.0907810540684385,
-4.01339304295351,
-7.83896330057243,
-12.9761628499025,
-15.9909644184075,
-20.1248772244162,
-23.9739257259396,


])



def plotaccuracy(targets1, targets2, ferror, berror, ferror2, berror2, error3, name):
    plt.figure(figsize=(8, 5))
    plt.plot(targets1, ferror, 'vr', label="Forward Error")
    plt.plot(targets1, berror, '^b', label="Forward Error")
    plt.plot(targets2, ferror2, '--r',label="xi forward")
    plt.plot(targets2, berror2, ':b', label="xi backward")
    plt.plot(targets2, error3, 'g-', label="xi")

    # Labels and legend
    plt.xlabel("Displacement")
    plt.ylabel("Error")
    plt.title(f"{name} Positioning Error")
    plt.axhline(0, color='black', linewidth=0.5)  # Horizontal reference
    plt.legend()
    plt.grid()

    # Display the plot
    plt.show()



plotaccuracy(targets1, targets2, ferror, berror, ferror2, berror2, error3, "Z")




#plotmovement(dx, dy, dz, thetax, thetay, thetaz, boxcorners, "True")
#plotmovement(dx, dyegg, dzegg, thetaxegg, thetayegg, thetazegg, boxcorners, "Angular Priority Scaled")
#plotmovement(dx, dyegg, dzegg, thetax, thetay, thetaz, boxcorners, "Straightness Priority Scaled")