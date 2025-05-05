# linearcarriagemap
A program developed to map out any points on a linear positioning carriage via csv files containing error data

htmpython.py - the main program
linpos.py - another work-in-progress program used to process linear positioning data in accordance with iso 230 standards

to use this-
make 6 folders in the same directory as htmpython.py- naming them dx, dy, dz, thetax, thetay, thetaz and upload corresponding csv data

the csv files in this case are laid out in 5 columns, Run #,Pos#,Target Value,Error Value,Out of spec.
