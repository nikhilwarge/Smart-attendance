##########################
#Import module
import tkinter.simpledialog as tsd
from tkinter import *
import tkinter as tk
from PIL import ImageTk,Image
from tkinter import messagebox as mess
import datetime
from imutils import paths
import csv
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import imutils
import pickle
import time
import cv2
################check all file present or not########
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
#check algorithm file is present or not
def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        #proced for further exucuation
        pass
    else:
        #show error message
        mess._show(title='Some file missing', message='Please contact us for help')
        window.destroy()
###################################################################################
#this will show contact details  when we click on help
def contact():
    mess._show(title='Contact us', message="Please contact us on : 'units2019@gmail.com' ")
###################################CLEAR BUTTON ###################################################
def clear():
    txt.delete(0, 'end')
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)
def clear2():
    txt2.delete(0, 'end')
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)
def clear3():
    txt3.delete(0, 'end')
#################save pass##########
# this function save pass ward
def save_pass():
    assure_path_exists("TrainingImageLabel/")
    # to check path is present or not
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    # check path  is present or not
    if exists1:
        tf = open("TrainingImageLabel\psd.txt", "r")
        # save the pass to file pass
        key = tf.read()  # read entered key
    else:
        master.destroy()
        # destroy window
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        # to save new pass
        if new_pas == None:  #
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
            # to show message
        else:
            tf = open("TrainingImageLabel\psd.txt", "w")
            # save password
            tf.write(new_pas)  # write new password
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return  # savev aal data for further use
    op = (old.get())
    newp = (new.get())
    nnewp = (nnew.get())
    if (op == key):
        # method to check ppassward is correct or  not
        if (newp == nnewp):
            # creat new passwawrd
            txf = open("TrainingImageLabel\psd.txt", "w")
            txf.write(newp)
        else:
            mess._show(title='Error', message='Confirm new password again!!!')
            return
    else:
        mess._show(title='Wrong Password', message='Please enter correct old password.')
        return
    mess._show(title='Password Changed', message='Password changed successfully!!')
    master.destroy()  # break this statement
##################################TO CHANGE PASSWARD#################################################
def change_pass():
    global master
    master = tk.Tk()
    #set gometry for window
    master.geometry("400x160")
    master.resizable(False,False)
    master.title("Change Password")
    master.configure(background="white")
    #set background white
    #this is all for lable 
    lbl4 = tk.Label(master,text='    Enter Old Password',bg='white',font=('times', 12, ' bold '))
    lbl4.place(x=10,y=10)
    global old
    old=tk.Entry(master,width=25 ,fg="black",relief='solid',font=('times', 12, ' bold '),show='*')
    old.place(x=180,y=10)
    lbl5 = tk.Label(master, text='   Enter New Password', bg='white', font=('times', 12, ' bold '))
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(master, width=25, fg="black",relief='solid', font=('times', 12, ' bold '),show='*')
    new.place(x=180, y=45)
    lbl6 = tk.Label(master, text='Confirm New Password', bg='white', font=('times', 12, ' bold '))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(master, width=25, fg="black", relief='solid',font=('times', 12, ' bold '),show='*')
    nnew.place(x=180, y=80)
    cancel=tk.Button(master,text="Cancel", command=master.destroy ,fg="black"  ,bg="red" ,height=1,width=25 , activebackground = "white" ,font=('times', 10, ' bold '))
    cancel.place(x=200, y=120)
    save1 = tk.Button(master, text="Save", command=save_pass, fg="black", bg="#3ece48", height = 1,width=25, activebackground="white", font=('times', 10, ' bold '))
    save1.place(x=10, y=120)
    master.mainloop()
#**************************
#############################MAIN PASSWARD WINDOW########################################################
global lines
def psw():
    #check folder is present or not
    assure_path_exists("TrainingImageLabel/")
    #if exixt save passward
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    if exists1:
        tf = open("TrainingImageLabel\psd.txt", "r")
        #to read pass from filw if already save
        key = tf.read()
    else:
        #to set new passward 
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel\psd.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    password = tsd.askstring('Password', 'Enter Password', show='*')
    if (password == key):
        file = open("StudentDetails\StudentDetails.csv")
        reader = csv.reader(file)
        l = len(list(reader))
        lines=int(l)
        lines=((lines-1)//2)
        res = "Profile Saved Successfully"
        message1.configure(text=res)
        message.configure(text='Total Registrations till now  : ' + str(lines))
        return lines
    elif (password == None):
        pass
    else:
        mess._show(title='Wrong Password', message='You have entered wrong password')
#*****************************************
##########handel tmim eoperation###########
def tick():
    time_string = time.strftime(' %H:%M:%S')
    clock.config(text=time_string)
    clock.after(200,tick)
def date_m():
    time_string = time.strftime('%Y-%m-%d  ')
    datef.config(text=time_string)
    datef.after(200,tick)
#*****************************

global Id
def TakeImages():
    check_haarcascadefile()
    # first check algo  file is present or not
    columns = ['SERIAL NO.', 'ID','NAME']
    # save data to databse in csv format in row and column
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    serial = 0
    exists = os.path.isfile("StudentDetails\StudentDetails.csv")
    # if path is exixt then updte data
    print("0")
    if exists:
        with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1  # update serial no
        serial = (serial // 2)
        print("1")
        csvFile1.close()
    else:
        with open("StudentDetails\StudentDetails.csv", 'w') as csvFile1:
            # a+ appenda data to last
            writer = csv.writer(csvFile1)
            print("2")
            writer.writerow(columns)
            serial = 1
        csvFile1.close()
    cascade = 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(cascade)
    Id = (txt.get())
    Name = (txt2.get())
    dataset = 'dataset'
    sub_data = Name
    path = os.path.join(dataset, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)
        print(sub_data)
    if ((Name.isalpha()) or (' ' in Name)):
        print("Starting video stream...")
        cam = cv2.VideoCapture(0)
        time.sleep(2.0)
        total = 0
        while total < 80:
            print(total)
            _, frame = cam.read()
            img = imutils.resize(frame, width=400)
            rects = detector.detectMultiScale(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
                minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                p = os.path.sep.join([path, "{}.png".format(
                    str(total).zfill(5))])
                cv2.imwrite(p, img)
                total += 1
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Taken for ID : " + Id
        row = [serial, Id, Name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text=res)
    else:
        if (Name.isalpha() == False):
            res = "Enter Correct name"
            message.configure(text=res)
def traning():
    def dl():
        dataset = "dataset"
        embeddingFile = "output/embeddings.pickle"  # initial name for embedding file
        embeddingModel = "openface_nn4.small2.v1.t7"  # initializing model for embedding Pytorch
        # initialization of caffe model for face detection
        prototxt = "model/deploy.prototxt"
        model = "model/res10_300x300_ssd_iter_140000.caffemodel"
        # loading caffe model for face detection
        # detecting face from Image via Caffe deep learning
        detector = cv2.dnn.readNetFromCaffe(prototxt, model)
        # loading pytorch model file for extract facial embeddings
        # extracting facial embeddings via deep learning feature extraction
        embedder = cv2.dnn.readNetFromTorch(embeddingModel)
        # gettiing image paths
        imagePaths = list(paths.list_images(dataset))
        # initialization
        knownEmbeddings = []
        knownNames = []
        total = 0
        conf = 0.5
        # we start to read images one by one to apply face detection and embedding
        for (i, imagePath) in enumerate(imagePaths):
            print("Processing image {}/{}".format(i + 1, len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]
            # converting image to blob for dnn face detection
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
            # setting input blob image
            detector.setInput(imageBlob)
            # prediction the face
            detections = detector.forward()
            if len(detections) > 0:
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]
                if confidence > conf:
                    # ROI range of interest
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]
                    if fW < 20 or fH < 20:
                        continue
                    # image ttracjo blob for face
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    # facial features embedder input image face blob
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()
                    knownNames.append(name)
                    knownEmbeddings.append(vec.flatten())
                    total += 1
        print("Embedding:{0} ".format(total))
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(embeddingFile, "wb")
        f.write(pickle.dumps(data))
        f.close()
        print("Process Completed")
    dl()
    def ml():
        # initilizing of embedding & recognizer
        embeddingFile = "output/embeddings.pickle"
        # New & Empty at initial
        recognizerFile = "output/recognizer.pickle"
        labelEncFile = "output/le.pickle"
        print("Loading face embeddings...")
        data = pickle.loads(open(embeddingFile, "rb").read())
        print("Encoding labels...")
        labelEnc = LabelEncoder()
        labels = labelEnc.fit_transform(data["names"])
        print("Training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)
        f = open(recognizerFile, "wb")
        f.write(pickle.dumps(recognizer))
        f.close()
        f = open(labelEncFile, "wb")
        f.write(pickle.dumps(labelEnc))
        f.close()
    ml()
    print("Model trained.")
def Show():
    import csv
    import tkinter
    root = tkinter.Tk()
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    sub=(txt3.get())
    root.title("Attendance of " + sub)
    root.configure(background='snow')
    with open("Attendance\\" + sub + "_" + date + ".csv", newline="") as file:
        reader = csv.reader(file)
        r = 0

        for col in reader:
            c = 0
            for row in col:
                # i've added some styling
                label = tkinter.Label(root, width=20, height=1, fg="black", font=('times', 15, ' bold '),
                                      bg="grey", text=row, relief=tkinter.RIDGE)
                label.grid(row=r, column=c)
                c += 1
            r += 1

def TrackImages():
    import numpy as np
    import imutils
    import pickle
    import time
    import cv2
    import pandas as pd
    sub=(txt3.get()) #get subject from user
    embeddingModel = "openface_nn4.small2.v1.t7"
    global embeddingFile
    embeddingFile = "output/embeddings.pickle"
    recognizerFile = "output/recognizer.pickle"
    labelEncFile = "output/le.pickle"
    conf = 0.5
    print("Loading face detector...")
    prototxt = "model/deploy.prototxt"
    model = "model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt, model)
    print("Loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(embeddingModel)
    recognizer = pickle.loads(open(recognizerFile, "rb").read())
    le = pickle.loads(open(labelEncFile, "rb").read())
    box = []
    print("Starting video stream...")
    cam = cv2.VideoCapture(0)
    time.sleep(2.0)
    global name
    cnt = 0
    total=0
    while True:
        _, frame = cam.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                          swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                global nme
                name = le.classes_[j]
                nme=[name]
                print("name is", name)
                text = "{}  : {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if (cv2.waitKey(1)==ord("q"))or(total==50):
            break
        total=total+1
        print("name is ", name)
        global timeStamp
        assure_path_exists("Attendance/")
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        exists = os.path.isfile("Attendance\\" + sub + "_" + date + ".csv")
        print("up to exist")
        if exists:
            nme = [name]
            dict = {'name': nme}
            df = pd.DataFrame(dict)
            df.to_csv("Attendance\\" + sub + "_" + date + ".csv", mode='a', header=False)
        else:
            nme = [name]
            print("ths is  else")
            # dictionary of lists
            dict = {'name': nme}
            df = pd.DataFrame(dict)
            # saving the dataframe
            df.to_csv("Attendance\\" + sub + "_" + date + ".csv")
        print("cnt=50", cnt)
        if cnt == 49:
            import pandas as pd
            df = pd.read_csv(r"Attendance\\" + sub + "_" + date + ".csv")
            df[~df.duplicated(subset=['name'])].to_csv("Attendance\\" + sub + "_" + date + ".csv")
            print("work")
            Show()
        cnt = cnt + 1
    cam.release()
    cv2.destroyAllWindows()
#####################GUI################
window = tk.Tk()
window.geometry("1280x720")
window.resizable(True,False)
window.title("UNITS-Attendance System")
canvas = Canvas(window, width =1280, height =1000)
canvas.pack()  
img= ImageTk.PhotoImage(Image.open("background.png"))
canvas.create_image(2, 20, anchor=NW, image=img)
frame1 = tk.Frame(window, bg="#55d4c1")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)
frame2 = tk.Frame(window, bg="#55d4c1")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)
message3 = tk.Label(window, text="Smart Attendance System" ,fg="white",bg="#262523" ,width=55 ,height=1,font=('times', 29, ' bold '))
message3.place(x=10, y=10)
frame3 = tk.Frame(window, bg="#55d4c1")
frame3.place(relx=0.52, rely=0.09, relwidth=0.21, relheight=0.04)
frame4 = tk.Frame(window, bg="#55d4c1")
frame4.place(relx=0.26, rely=0.09, relwidth=0.31, relheight=0.04)
datef = tk.Label(frame4, fg="orange",bg="#262523" ,width=20 ,height=1,font=('times', 15, ' bold '))
datef.pack(fill='both',expand=1)
date_m()
clock = tk.Label(frame3,fg="orange",bg="#262523" ,width=20 ,height=1,font=('times', 15, ' bold '))
clock.pack(fill='both',expand=1)
tick()
head2 = tk.Label(frame2, text="                       For New Registrations                       ", fg="black",bg="#e3b781" ,font=('times', 17, ' bold ') )
head2.grid(row=5,column=2)
head1 = tk.Label(frame1, text="                       For Already Registered                       ", fg="black",bg="#e3b781" ,font=('times', 17, ' bold ') )
head1.place(x=0,y=0)
lbl = tk.Label(frame2, text="Enter ID",width=10  ,height=1  ,fg="black"  ,bg="#55d4c1" ,font=('times', 17, ' bold ') )
lbl.place(x=80, y=55)
txt = tk.Entry(frame2,width=32 ,fg="black",font=('times', 15, ' bold '))
txt.place(x=30, y=88)
lbl2 = tk.Label(frame2, text="Enter Name",width=10  ,fg="black"  ,bg="#55d4c1" ,font=('times', 17, ' bold '))
lbl2.place(x=80, y=140)
txt2 = tk.Entry(frame2,width=32 ,fg="black",font=('times', 15, ' bold ')  )
txt2.place(x=30, y=173)
message1 = tk.Label(frame2, text="1)Take Images  >>>  2)Save Profile" ,bg="#55d4c1" ,fg="black"  ,width=39 ,height=1, activebackground = "yellow" ,font=('times', 15, ' bold '))
message1.place(x=7, y=230)
message = tk.Label(frame2, text="" ,bg="#00aeff" ,fg="black"  ,width=39,height=1, activebackground = "yellow" ,font=('times', 16, ' bold '))
message.place(x=7, y=450)
lbl4 = tk.Label(frame1, text="Subject",width=10  ,fg="black"  ,bg="#55d4c1"  ,height=1 ,font=('times', 17, ' bold '))
lbl4.place(x=0, y=100)
lbl5 = tk.Label(frame1, text="Instructions:",width=20  ,fg="black"  ,bg="#55d4c1"  ,height=1 ,font=('times', 17, ' bold '))
lbl5.place(x=100, y=170)
lbl6 = tk.Label(frame1, text="1]Check studentDaitels file is close or not",width=40  ,fg="black"  ,bg="#55d4c1"  ,height=1 ,font=('times', 12, ' bold '))
lbl6.place(x=18, y=220)
lbl7 = tk.Label(frame1, text="2]train model once after completation of rgistration ",width=40  ,fg="black"  ,bg="#55d4c1"  ,height=1 ,font=('times', 12, ' bold '))
lbl7.place(x=50, y=270)
lbl8 = tk.Label(frame1, text="3]Attendance sheet store in attendance folder",width=40  ,fg="black"  ,bg="#55d4c1"  ,height=1 ,font=('times', 12, ' bold '))
lbl8.place(x=30, y=320)
lbl9 = tk.Label(frame1, text="4]Traning time is depend upon data set ",width=40  ,fg="black"  ,bg="#55d4c1"  ,height=1 ,font=('times', 12, ' bold '))
lbl9.place(x=10, y=370)
txt3 = tk.Entry(window, width=32, bg="yellow", fg="red", font=('times', 15, ' bold '))
txt3.place(x=170, y=252)
res=0
exists = os.path.isfile("StudentDetails\StudentDetails.csv")
if exists:
    with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for l in reader1:
            res = res + 1
    res = (res // 2) - 1
    csvFile1.close()
else:
    res = 0
message.configure(text='Total Registrations till now  : '+str(res),bg="#55d4c1")
##################### MENUBAR #################################
menubar = tk.Menu(window,relief='ridge')
filemenu = tk.Menu(menubar,tearoff=0)
filemenu.add_command(label='Change Password', command = change_pass)
filemenu.add_command(label='Contact Us', command = contact)
filemenu.add_command(label='Exit',command = window.destroy)
menubar.add_cascade(label='Help',font=('times', 29, ' bold '),menu=filemenu)
clearButton = tk.Button(frame2, text="Clear", command=clear  ,fg="black"  ,bg="#ea2a2a"  ,width=11 ,activebackground = "white" ,font=('times', 11, ' bold '))
clearButton.place(x=335, y=86)
clearButton2 = tk.Button(frame2, text="Clear", command=clear2  ,fg="black"  ,bg="#ea2a2a"  ,width=11 , activebackground = "white" ,font=('times', 11, ' bold '))
clearButton2.place(x=335, y=172)
clearButton3 = tk.Button(frame1, text="Clear", command=clear3  ,fg="black"  ,bg="#ea2a2a"  ,width=11 ,activebackground = "white" ,font=('times', 11, ' bold '))
clearButton3.place(x=350, y=130)
takeImg = tk.Button(frame2, text="Take Images", command=TakeImages  ,fg="white"  ,bg="blue"  ,width=34  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
takeImg.place(x=30, y=300)
trainImg = tk.Button(frame2, text="Save Profile", command=psw ,fg="white"  ,bg="blue"  ,width=34  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
trainImg.place(x=30, y=380)
trackImg = tk.Button(frame1, text="Take Attendance", command=TrackImages  ,fg="black"  ,bg="yellow"  ,width=35  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
trackImg.place(x=30,y=50)
trackImg1 = tk.Button(frame1, text="Train Model", command=traning  ,fg="black"  ,bg="yellow"  ,width=35  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
trackImg1.place(x=30,y=420)
quitWindow = tk.Button(frame1, text="Quit", command=window.destroy  ,fg="black"  ,bg="red"  ,width=35 ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
quitWindow.place(x=30, y=500)
##################### END ######################################
window.configure(menu=menubar)
window.mainloop()
#######################thank u so much###############
