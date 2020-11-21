import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import tree
from subprocess import call
from sklearn.neighbors import KNeighborsClassifier
from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog
import pylab as pl
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve, auc
import csv
import tkMessageBox
# Importing dataset
data = pd.read_csv("Attribute DataSet.csv")
X=[]
y=[]
root = Tk()
root.geometry("1800x800")

left = Frame(bg="misty rose",width=800, height=800, borderwidth=2, relief="solid")
right = Frame(bg="misty rose",width=800, height=800, borderwidth=2, relief="solid")

left.pack(side="left", expand=True, fill="both")
right.pack(side="right", expand=True, fill="both")

fig = Figure(figsize=(6,4))
canvas = FigureCanvasTkAgg(fig, master=left)
fig1 = Figure(figsize=(6,4))
canvas1 = FigureCanvasTkAgg(fig1, master=left)
pl1=fig1.add_subplot(111)
w=Label()
"""itemsforlistbox=[ "Style_cleaned",
	    "Price_cleaned",
	    "Rating",
	    "Size_cleaned",
	    "Season_cleaned",
	    "NeckLine_cleaned",
	    "SleeveLength_cleaned",
	    "waiseline_cleaned",
	    "Material_cleaned",
	    "FabricType_cleaned",
	    "Decoration_cleaned",
	    "PatternType_cleaned"]"""
itemsforlistbox=[ "Style",
	    "Price",
	    "Rating",
	    "Size",
	    "Season",
	    "NeckLine",
	    "SleeveLength",
	    "waiseline",
	    "Material",
	    "FabricType",
	    "Decoration",
	    "PatternType"]
selected=[]
def remove() :
	global selected
	selected=[]
def CurSelet(event):
	widget = event.widget
	selection=widget.curselection()
	picked = widget.get(selection[0])
	if picked not in selected :
		selected.append(picked)
	print(selected)
mylistbox=Listbox(right,width=30,height=15,font=('times',13))
mylistbox.bind('<<ListboxSelect>>',CurSelet)
mylistbox.place(x=150,y=350)
for items in itemsforlistbox:
    mylistbox.insert(END,items)
def prep():
	global data,X,y,button1,button2,button3,button4,button5
	data.drop('Color',axis=1,inplace=True)	
	# Cleaning dataset of NaN
	#Style, Price, Rating, Size, Season, NeckLine, SleeveLength, waiseline, Material, FabricType, Decoration, Pattern, Type, Recommendation
	data=data[[
	    "Style",
	    "Price",
	    "Rating",
	    "Size",
	    "Season",
	    "NeckLine",
	    "SleeveLength",
	    "waiseline",
	    "Material",
	    "FabricType",
	    "Decoration",
	    "Pattern Type",
	    "Recommendation"
	]].dropna(axis=0, how='any')
	#print(data)
	# Convert categorical variable to numeric
	data["Style_cleaned"]=np.where(data["Style"]=="Bohemia",0,
					np.where(data["Style"]=="brief",1,
						np.where(data["Style"]=="casual",2,
							np.where(data["Style"]=="cute",3,
								np.where(data["Style"]=="fashion",4,
									np.where(data["Style"]=="flare",5,
										np.where(data["Style"]=="novelty",6,
											np.where(data["Style"]=="OL",7,
												np.where(data["Style"]=="party",8,
													np.where(data["Style"]=="sexy",9,
														np.where(data["Style"]=="vintage",10,
															np.where(data["Style"]=="work",11,12)
															)
														)
													)
												)
											)		
										)
									)
								)
							)
						)
					)
	data["Price_cleaned"]=np.where(data["Price"]=="Low",0,
		                          np.where(data["Price"]=="Average",1,
		                                   np.where(data["Price"]=="Medium",2,
								np.where(data["Price"]=="High",3,
									np.where(data["Price"]=="Very-High",4,5)
									)
							)
		                                  )
		                         )
	data["Size_cleaned"]=np.where(data["Size"]=="S",0,
		                          np.where(data["Size"]=="M",1,
		                                   np.where(data["Size"]=="L",2,
								np.where(data["Size"]=="XL",3,
									np.where(data["Size"]=="Free",4,5)
									)
							)
		                                  )
		                         )
	data["Season_cleaned"]=np.where(data["Season"]=="Autumn",0,
		                          np.where(data["Season"]=="Winter",1,
		                                   np.where(data["Season"]=="Spring",2,
								np.where(data["Season"]=="Summer",3,4)
									)
							)
		                                  )
		                         
	data["NeckLine_cleaned"]=np.where(data["NeckLine"]=="O-neck",0,
					np.where(data["NeckLine"]=="backless",1,
						np.where(data["NeckLine"]=="board-neck",2,
							np.where(data["NeckLine"]=="Bowneck",3,
								np.where(data["NeckLine"]=="halter",4,
									np.where(data["NeckLine"]=="mandarin-collor",5,
										np.where(data["NeckLine"]=="open",6,
											np.where(data["NeckLine"]=="peterpan-collor",7,
												np.where(data["NeckLine"]=="ruffled",8,
													np.where(data["NeckLine"]=="scoop",9,
														np.where(data["NeckLine"]=="slash-neck",10,
															np.where(data["NeckLine"]=="square-collar",11,
																np.where(data["NeckLine"]=="sweetheart",12,
																	np.where(data["NeckLine"]=="turndowncollar",13,
																		np.where(data["NeckLine"]=="V-neck",14,
	15)
																)
															)
														)
													)
												)
											)		
										)
									)
								)
							)
						)
					)
				)
			)
	data["SleeveLength_cleaned"]=np.where(data["SleeveLength"]=="full",0,
					np.where(data["SleeveLength"]=="half",1,
						np.where(data["SleeveLength"]=="halfsleeves",2,
							np.where(data["SleeveLength"]=="butterfly",3,
								np.where(data["SleeveLength"]=="sleveless",4,
									np.where(data["SleeveLength"]=="short",5,
										np.where(data["SleeveLength"]=="threequarter",6,
											np.where(data["SleeveLength"]=="turndown",7,8)
											)		
										)
									)
								)
							)
						)
					)
	data["waiseline_cleaned"]=np.where(data["waiseline"]=="dropped",0,
		                          np.where(data["waiseline"]=="empire",1,
		                                   np.where(data["waiseline"]=="natural",2,
								np.where(data["waiseline"]=="princess",3,4)
							)
		                                  )
		                         )
	data["Material_cleaned"]=np.where(data["Material"]=="wool",0,
		                          np.where(data["Material"]=="cotton",1,
		                                   np.where(data["Material"]=="mix",2,3)
		                                  )
		                         )
	data["FabricType_cleaned"]=np.where(data["FabricType"]=="shafoon",0,
					np.where(data["FabricType"]=="dobby",1,
						np.where(data["FabricType"]=="popline",2,
							np.where(data["FabricType"]=="satin",3,
								np.where(data["FabricType"]=="knitted",4,
									np.where(data["FabricType"]=="jersey",5,
										np.where(data["FabricType"]=="flannel",6,
											np.where(data["FabricType"]=="corduroy",7,8)
											)		
										)
									)
								)
							)
						)
					)
	data["Decoration_cleaned"]=np.where(data["Decoration"]=="applique",0,
					np.where(data["Decoration"]=="beading",1,
						np.where(data["Decoration"]=="bow",2,
							np.where(data["Decoration"]=="button",3,
								np.where(data["Decoration"]=="cascading",4,
									np.where(data["Decoration"]=="crystal",5,
										np.where(data["Decoration"]=="draped",6,
											np.where(data["Decoration"]=="embroridary",7,
												np.where(data["Decoration"]=="feathers",8,
													np.where(data["Decoration"]=="flowers",9,10)
													)
												)
											)		
										)
									)
								)
							)
						)
					)
	data["PatternType_cleaned"]=np.where(data["Pattern Type"]=="solid",0,
		                          np.where(data["Pattern Type"]=="animal",1,
		                                   np.where(data["Pattern Type"]=="dot",2,
								np.where(data["Pattern Type"]=="leapard",3,4)
							)
		                                  )
		                         )

	data_cleaned=np.array([data.Style_cleaned,data.Price_cleaned,data.Rating,data.Size_cleaned,data.Season_cleaned,data.NeckLine_cleaned,data.SleeveLength_cleaned,data.waiseline_cleaned,data.Material_cleaned,data.FabricType_cleaned,data.Decoration_cleaned,data.PatternType_cleaned,data.Recommendation])
	data_cleaned=data_cleaned.T
	myFile = open('cleaned.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerow([ "Style",
	    "Price",
	    "Rating",
	    "Size",
	    "Season",
	    "NeckLine",
	    "SleeveLength",
	    "waiseline",
	    "Material",
	    "FabricType",
	    "Decoration",
	    "PatternType",
	    "Recommendation"])
	    for i in data_cleaned :
		writer.writerow(i)
		
	print("Writing complete")
	button5.config(state=DISABLED)
	button1.config(state=NORMAL)
	button2.config(state=NORMAL)
	button3.config(state=NORMAL)
	
	y = np.array(data.Recommendation)



def NB():
	global fig,selected
	global canvas
	global fig1
	global canvas1
	global pl1,w
	global data,X,y,button1,button2,button3,button4,itemsforlistbox
	X=[]
	X=pd.read_csv("cleaned.csv")
	for i in itemsforlistbox :
		if i not in selected :
			X.drop(i,axis=1,inplace=True)
	X=np.array(X)	
	conf_mat_NB=[[0,0],[0,0]]	
	print("NB")
	kf = KFold(n_splits=10)
	KFold(n_splits=10, random_state=None, shuffle=False)
	a=0
	gnb = GaussianNB()
	for train_index, test_index in kf.split(X):
	    print("TRAIN:", train_index, "TEST:", test_index)
	    X_train, X_test = X[train_index], X[test_index]
	    y_train, y_test = y[train_index], y[test_index]	    
	    gnb.fit(X_train,y_train)
	    y_pred=gnb.predict(X_test)
	    cm=confusion_matrix(y_test, y_pred)
	    conf_mat_NB=conf_mat_NB+cm
	    a+=metrics.accuracy_score(y_test,y_pred)
	sum_NB=a/10
	print(sum_NB)
	print(conf_mat_NB)
	fig = Figure(figsize=(6,4))
	fig1 = Figure(figsize=(6,4))
	pl1=fig1.add_subplot(111)
	pl=fig.add_subplot(111)
	plt.clf()
	pl.matshow(cm)
	#pl.title('Confusion matrix of the classifier')
	#pl.colorbar()
	#pl.show()
	labels = ['trending', 'not trending']
	
	plt.title('Confusion matrix of the classifier')
	pl.set_xticklabels([''] + labels)
	pl.set_yticklabels([''] + labels)
	plt.xlabel('Predicted')
	plt.ylabel('True')
	#plt.show()
	s=[['TP','FP'],['FN','TN']]
	for i in range(2):
		for j in range(2):
			pl.text(j,i,str(s[i][j])+"= "+str(conf_mat_NB[i][j]))
	canvas = FigureCanvasTkAgg(fig, master=left)
        canvas.get_tk_widget().pack()
        canvas.draw()
	
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	plt.title('Receiver Operating Characteristic')
	pl1.plot(false_positive_rate, true_positive_rate, 'b',
	label='AUC = %0.2f'% roc_auc)
	plt.legend(loc='lower right')
	pl1.plot([0,1],[0,1],'r--')
	plt.xlim([-0.1,1.2])
	plt.ylim([-0.1,1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	#plt.show()
	canvas1 = FigureCanvasTkAgg(fig1, master=left)
        canvas1.get_tk_widget().pack()
        canvas1.draw()
	str1=str(sum_NB)
	w=Label(right,width=15,height=2,text='Accuracy = '+str1 )
	w.pack()
	w.place(x=250,y=300)
	button1.config(state=DISABLED)
	button2.config(state=DISABLED)
	button3.config(state=DISABLED)
	button4.config(state=NORMAL)
	
def DTree():
	global fig
	global canvas
	global fig1
	global canvas1
	global pl1,w
	global data,X,y,button1,button2,button3,button4,itemsforlistbox
	X=[]
	X=pd.read_csv("cleaned.csv")
	for i in itemsforlistbox :
		if i not in selected :
			X.drop(i,axis=1,inplace=True)
	X=np.array(X)
	print(X)
	conf_mat_tree=[[0,0],[0,0]]	
	print("Decision Tree")
	kf = KFold(n_splits=10)
	KFold(n_splits=10, random_state=None, shuffle=False)
	a=0
	model = tree.DecisionTreeClassifier()
	for train_index, test_index in kf.split(X):
	    print("TRAIN:", train_index, "TEST:", test_index)
	    X_train, X_test = X[train_index], X[test_index]
	    y_train, y_test = y[train_index], y[test_index]
	    model.fit(X_train,y_train)
	    y_pred=model.predict(X_test)
	    cm=confusion_matrix(y_test, y_pred)
	    conf_mat_tree=conf_mat_tree+cm
	    a+=metrics.accuracy_score(y_test,y_pred)
	    tree.export_graphviz(model, out_file='tree.dot')
	    call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree'+str(train_index)+'.png'])
	sum_tree=a/10
	print(sum_tree)
	print(conf_mat_tree)
	
	fig = Figure(figsize=(6,4))
	fig1 = Figure(figsize=(6,4))
	pl1=fig1.add_subplot(111)	
	pl=fig.add_subplot(111)
	plt.clf()
	pl.matshow(cm)
	#pl.title('Confusion matrix of the classifier')
	#pl.colorbar()
	#pl.show()
	labels = ['trending', 'not trending']
	
	plt.title('Confusion matrix of the classifier')
	pl.set_xticklabels([''] + labels)
	pl.set_yticklabels([''] + labels)
	plt.xlabel('Predicted')
	plt.ylabel('True')
	#plt.show()
	s=[['TP','FP'],['FN','TN']]
	for i in range(2):
		for j in range(2):
			pl.text(j,i,str(s[i][j])+"= "+str(conf_mat_tree[i][j]))
       	canvas = FigureCanvasTkAgg(fig, master=left)
        canvas.get_tk_widget().pack()
        canvas.draw()
	
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	plt.title('Receiver Operating Characteristic')
	pl1.plot(false_positive_rate, true_positive_rate, 'b',
	label='AUC = %0.2f'% roc_auc)
	plt.legend(loc='lower right')
	pl1.plot([0,1],[0,1],'r--')
	plt.xlim([-0.1,1.2])
	plt.ylim([-0.1,1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	#plt.show()
	canvas1 = FigureCanvasTkAgg(fig1, master=left)
        canvas1.get_tk_widget().pack()
        canvas1.draw()
	str1=str(sum_tree)
	w=Label(right,width=15,height=2,text='Accuracy = '+str1 )
	w.pack()
	w.place(x=250,y=300)
	
	button1.config(state=DISABLED)
	button2.config(state=DISABLED)
	button3.config(state=DISABLED)
	button4.config(state=NORMAL)
def KNN():
	global fig
	global canvas
	global fig1
	global canvas1
	global pl1,w
	global data,X,y,button1,button2,button3,button4,itemsforlistbox
	X=[]
	X=pd.read_csv("cleaned.csv")
	for i in itemsforlistbox :
		if i not in selected :
			X.drop(i,axis=1,inplace=True)
	X=np.array(X)
	print(X)
	conf_mat_knn=[[0,0],[0,0]]
	print("KNN")
	kf = KFold(n_splits=10)
	KFold(n_splits=10, random_state=None, shuffle=False)
	a=0
	knn = KNeighborsClassifier(n_neighbors=2)
	for train_index, test_index in kf.split(X):
	    print("TRAIN:", train_index, "TEST:", test_index)
	    X_train, X_test = X[train_index], X[test_index]
	    y_train, y_test = y[train_index], y[test_index]
	    # Train the model using the training sets
	    knn.fit(X_train,y_train)
	    y_pred=knn.predict(X_test)
	    cm=confusion_matrix(y_test, y_pred)
	    conf_mat_knn=conf_mat_knn+cm
	    a+=metrics.accuracy_score(y_test,y_pred)
	sum_knn=a/10
	print(sum_knn)
	print(conf_mat_knn)
	fig = Figure(figsize=(6,4))
	fig1 = Figure(figsize=(6,4))
	pl1=fig1.add_subplot(111)	
	pl=fig.add_subplot(111)
	plt.clf()
	pl.matshow(cm)
	#pl.title('Confusion matrix of the classifier')
	#pl.colorbar()
	#pl.show()
	labels = ['trending', 'not trending']
	
	plt.title('Confusion matrix of the classifier')
	pl.set_xticklabels([''] + labels)
	pl.set_yticklabels([''] + labels)
	plt.xlabel('Predicted')
	plt.ylabel('True')
	#plt.show()
	s=[['TP','FP'],['FN','TN']]
	for i in range(2):
		for j in range(2):
			pl.text(j,i,str(s[i][j])+"= "+str(conf_mat_knn[i][j]))
       	canvas = FigureCanvasTkAgg(fig, master=left)
        canvas.get_tk_widget().pack()
        canvas.draw()
	
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	plt.title('Receiver Operating Characteristic')
	pl1.plot(false_positive_rate, true_positive_rate, 'b',
	label='AUC = %0.2f'% roc_auc)
	plt.legend(loc='lower right')
	pl1.plot([0,1],[0,1],'r--')
	plt.xlim([-0.1,1.2])
	plt.ylim([-0.1,1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	#plt.show()
	canvas1 = FigureCanvasTkAgg(fig1, master=left)
        canvas1.get_tk_widget().pack()
        canvas1.draw()
	str1=str(sum_knn)
	w=Label(right,width=15,height=2,text='Accuracy = '+str1 )
	w.pack()
	w.place(x=250,y=300)
	
	button1.config(state=DISABLED)
	button2.config(state=DISABLED)
	button3.config(state=DISABLED)
	button4.config(state=NORMAL)
def Clear():
	global canvas
	global canvas1
	global pl1,w
	global data,X,y,button1,button2,button3,button4
	pl1.cla()
	canvas.get_tk_widget().destroy()
	canvas1.get_tk_widget().destroy()
	w.destroy()
	button1.config(state=NORMAL)
	button2.config(state=NORMAL)
	button3.config(state=NORMAL)
	button4.config(state=DISABLED)
#front end
button1=Button(right,activebackground="white",activeforeground="red",bg="salmon1",fg="black",text="Naive Bayes", command=NB,height=3,width=10)
button2=Button(right,activebackground="white",activeforeground="red",bg="salmon1",fg="black",text="Decision Tree", command=DTree,height=3,width=10)
button3=Button(right,activebackground="white",activeforeground="red",bg="salmon1",fg="black",text="KNN", command=KNN,height=3,width=10)
button4=Button(right,activebackground="white",activeforeground="red",bg="salmon1",fg="black",text="Clear", command=Clear,height=3,width=10)
button5=Button(right,activebackground="white",activeforeground="red",bg="salmon1",fg="black",text="Preprocessing", command=prep,height=3,width=10)
button6=Button(right,activebackground="white",activeforeground="red",bg="salmon1",fg="black",text="Add to List",height=3,width=10)
button7=Button(right,activebackground="white",activeforeground="red",bg="salmon1",fg="black",text="Empty List", command=remove,height=3,width=10)
button1.config(state=DISABLED)
button2.config(state=DISABLED)
button3.config(state=DISABLED)
button4.config(state=DISABLED)
button1.pack(pady=10)
button2.pack(pady=10)
button3.pack(pady=10)
button4.pack(pady=10)
button5.pack(pady=10)
button6.pack(pady=10)
button7.pack(pady=10)
button1.place(x=30,y=50)
button2.place(x=180,y=50)
button3.place(x=330,y=50)
button4.place(x=30,y=150)
button5.place(x=180,y=150)
button6.place(x=450,y=350)
button7.place(x=450,y=450)

def openFile():
	right.filename = tkFileDialog.askopenfilename(initialdir = "/home",title = "Select file",filetypes = (("arff files","*.txt"),("all files","*.*")))
	print (right.filename) 


root.mainloop()

