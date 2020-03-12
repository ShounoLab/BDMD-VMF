Data were collected with next sensors setting:
Range of the gyroscopes from -2000 to 2000 deg/sec.
Range of the accelerometers from -2g to 2g. Where g is gravity acceleration
Values of the gyroscopes and the accelerometers encoded by int_16 datatype.
Values of the EMGs encoded by uint_8 datatype.

Here is table with information about recorded activities:

ID		Activity 			Time (min) Percent 	Description
1		walking 			192 		32.15 	Walking and turning at various speed on flat surface.
2		running 			20 			3.39 	Running at various pace.
3		going_up 			37 			6.23 	Taking stairs up at various speed.
4		going_down 			33 			5.52 	Taking the stairs down at various speed and steps.
5		sitting 			68 			11.45 	Sitting on a chair; sitting on floor not included.
6		sitting down 		6 			1.14 	Sitting on a chair; sitting down on floor not included.
7		standing up 		6 			1.06 	Standing up from chair.
8		standing 			93 			15.56 	Static standing on solid surface.
9		bicycling 			44 			7.41 	Regular bicycling.
10		up_by_elevator 		25 			4.22 	Standing in elevator while moving up.
11		down_by_elevator 	19 			3.30 	Standing in elevator while moving down.
12		sitting in car 		51 			8.55 	Sitting while traveling by car. 
		Total 				598 		100.00 

Here is table with information about participants:

id	weight (kg)	height (cm)	age	sex (M=Male, F=Female)
1 	75 			177 		24 	M 
2 	80 			183 		22 	M 
3 	65 			183 		23 	M
4 	93 			189 		24 	M
5 	63 			183 		35 	M
6 	54 			168 		25 	F
7 	52 			161 		22 	F		
8 	80 			176 		23 	M
9 	65 			175 		24 	F
10 	118 		183 		27 	M
11 	85 			203 		24 	M
12 	85 			192 		23 	M
13 	64 			174 		18 	M
14 	68 			175 		19 	M
15 	72 			178 		23 	M
16 	48 			164 		26 	F
17 	85 			179 		25 	M
18 	70 			180 		19 	M

Every file name was created according to the following template HGD_vX_ACT_PR_CNT.txt. Here is table with description of the file naming convention:

TAG 	Description 	Type	Comment
HGD 	Prefix 			fixed 	Data files start with this prefix 
vX 		Version number 	integer Indicates the version of the data format
ACT		Activity 		string 	Indicates the type of activity
PR 		Participant ID 	integer Indicates the subject whos data was recorded
CNT		Counter 		integer Counter for repeated experiments	
		
For example, a file named HGD_v1_walking_17_02.txt, contains data from participant 17 while he was walking repeated for the second time.
Each file contains header. Here is table with description of the data file header:

TAG 		Description 					Type				Comment
#Activity 	List of the activities 			string 				lists the activity names in this file
#ActivityID List of the ID of activities 	list of integers 	lists the activity IDs in this file
#Date-Time 	Date and Time 					MM-DD-HR-MN 		Month-Day-Hour-Min format
