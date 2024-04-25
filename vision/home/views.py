from django.shortcuts import render,redirect
from. models import *
from django.contrib import messages
import pyttsx3

# Create your views here.
def home(request):
    return render(request,'index.html')

def log(request):
    if request.method=="POST":
        try:
            Email=request.POST.get('Email')
            password=request.POST.get('Password')
            log=c_reg.objects.get(Email=Email,Password=password)
            request.session['F_name']=log.F_name
            request.session['id']=log.id
            return redirect('home1')
        except c_reg.DoesNotExist as e:
            messages.info(request,'Invalid User')
    return render(request,'login.html')

def reg(request):
    if request.method=="POST":
        F_name=request.POST.get('F_name')
        Email=request.POST.get('Email')
        Password=request.POST.get('Password')
        Cornfirm_password=request.POST.get('Confirm_password')
        
        if Password==Cornfirm_password:
            if c_reg.objects.filter(Email=Email).exists():
                messages.info(request,'Email Adress Already Exists')
            else:
                reg=c_reg(F_name=F_name,Email=Email,Password=Password)
                reg.save()
                return redirect('log')
        else:
            messages.info(request,'Password not match')
    return render(request,'Register.html')

def home1(request):
    return render(request,'home1.html')

def face_verify(request):
    id=request.session['id']
    data=c_reg.objects.get(id=id)
    n=data.F_name
    pred=25
 
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    if request.method=="POST":
        import numpy as np
        from . import PrepareDataset
        # from . PrepareDataset import *
        # import AddNewFace 
        import cv2
        import os
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV, KFold
        import pickle


        images = []
        labels = []
        labels_dic = {}

        # choice = input("Do you want to add new face? (Yes or No) ")
        # if choice == 'yes':
        #     AddNewFace.add_face()


        def collect_dataset():

            people = [person for person in os.listdir("C:\\vision\\vision\\main project\\vision\\people")]

            for i, person in enumerate(people):
                labels_dic[i] = person
                for image in os.listdir("people/" + person):
                    if image.endswith('.jpg'):
                        images.append(cv2.imread("people/" + person + '/' + image, 0))
                        labels.append(i)
            return images, np.array(labels), labels_dic


        images, labels, labels_dic = collect_dataset()

        X_train = np.asarray(images)
        train = X_train.reshape(len(X_train), -1)

        sc = StandardScaler()
        X_train_sc = sc.fit_transform(train.astype(np.float64))
        pca1 = PCA(n_components=.97)
        new_train = pca1.fit_transform(X_train_sc)
        kf = KFold(n_splits=5,shuffle=True)
        param_grid = {'C': [.0001, .001, .01, .1, 1, 10]}
        gs_svc = GridSearchCV(SVC(kernel='linear', probability=True), param_grid=param_grid, cv=kf, scoring='accuracy')
        gs_svc.fit(new_train, labels)
        clf = gs_svc.best_estimator_
        filename = 'svc_linear_face.pkl'
        f = open(filename, 'wb')
        pickle.dump(clf, f)
        f.close()

        filename = 'svc_linear_face.pkl'
        svc1 = pickle.load(open(filename, 'rb'))

        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.namedWindow("opencv_face ", cv2.WINDOW_AUTOSIZE)
        count=0
        bool=True
        while bool:
            ret, frame = cam.read()

            faces_coord = detect_face(frame)  # detect more than one face
            if len(faces_coord):
                faces = normalize_faces(frame, faces_coord)

                for i, face in enumerate(faces):  # for each detected face

                    t = face.reshape(1, -1)
                    t = sc.transform(t.astype(np.float64))
                    test = pca1.transform(t)
                    prob = svc1.predict_proba(test)
                    confidence = svc1.decision_function(test)
                    # print(confidence)
                    # print(prob)

                    pred = svc1.predict(test)
                    print(pred, pred[0])
                    print("Predicted=",pred)


                    name = labels_dic[pred[0]].capitalize()
                    print(name)
                    print("name=",name)
                    if name=="Unknown":
                        print("Not recognized person")
                        # cv2.putText(frame, name,"Not recozied person" )
                        cv2.putText(frame, "Not recognized person", (6, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (55, 55, 243), 2,
                        cv2.LINE_AA)
                        
                        engine.say("Not recognized person")
                        engine.runAndWait()
                        
                    else:
                        print(" recognized person")
                        cv2.putText(frame, " recognized person", (6, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (55, 55, 243), 2,
                        cv2.LINE_AA)
                        
                        engine.say("Recognized person: " + name)
                        engine.runAndWait()

                    

                        cv2.putText(frame, name, (faces_coord[i][0], faces_coord[i][1] - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (66, 53, 243), 2)

                draw_rectangle(frame, faces_coord)  # rectangle around face

            # cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2,
            #             cv2.LINE_AA)
            

            cv2.imshow("opencv_face", frame)  # live feed in external
            Pred=pred+1
            if int(Pred) == int(id):
                count=count+1
                # print(count)
                # print(voteid)
            if count>20:
                bool=False
                return redirect("home1")
           

            if cv2.waitKey(5) == 27:
                break
            

        cam.release()
        cv2.destroyAllWindows()
    return render(request,'fverify.html')
import numpy as np
# from PrepareDataset import *
from . PrepareDataset import *

        # from . PrepareDataset import *
        # import AddNewFace 
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
import pickle

def addface(request):
        global Name
        if request.method=="POST":
            F_name=request.POST.get('Name')
            id=request.session['id']
            choice = 'yes'
            cam = cv2.VideoCapture(0)
            folder = "people/" + str(F_name)


            
            try:
                os.mkdir(folder)

                flag_start_capturing = False
                sample = 1
                cv2.namedWindow("Face", cv2.WINDOW_AUTOSIZE)
                while True:
                    ret, frame = cam.read()
                    if frame.ndim != 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    detector = cv2.CascadeClassifier("xml/frontal_face.xml")
                    faces = detector.detectMultiScale(frame, 1.2, 5)
                    faces_coord=faces

                    if len(faces_coord):
                        faces = normalize_faces(frame, faces_coord)
                        cv2.imwrite(folder + '/' + str(sample) + '.jpg', faces[0])

                                # if flag_start_capturing:
                        sample += 1

                    draw_rectangle(frame, faces_coord)
                    cv2.imshow("Face", frame)
                    keypress = cv2.waitKey(1)

                    if keypress == ord('c'):

                        if not flag_start_capturing:
                            flag_start_capturing = True

                    if sample > 150:
                        return redirect("home1")
                cam.release()
                cv2.destroyAllWindows()   

            except FileExistsError:
                    print("Already exists")

        return render(request,'addface.html')







# *********************************Object Detection*************************



def objdec(request):
        from ultralytics import YOLO
        import cv2
        import math     
        import pyttsx3  # Import the text-to-speech library
        engine = pyttsx3.init()

# start webcam
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        # model
        model = YOLO("yolo-Weights/yolov8n.pt")

        # object classes
        classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                    "teddy bear", "hair drier", "toothbrush","pen"
                    ]


        while True:
            success, img = cap.read()
            results = model(img, stream=True)

            # List to store detected object names
            detected_objects = []

            # coordinates
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    # confidence
                    confidence = math.ceil((box.conf[0]*100))/100

                    # class name
                    cls = int(box.cls[0])
                    class_name = classNames[cls]
                    detected_objects.append(class_name)

                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(img, class_name, org, font, fontScale, color, thickness)

            # Convert detected object names to a single string
            detected_objects_str = ", ".join(detected_objects)
            
            # Speak out the detected object names
            engine.say(detected_objects_str)
            engine.runAndWait()

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



        import torch
        from ultralytics import YOLO

        # Load the YOLO model
        model = YOLO("yolo-Weights/yolov8n.pt")

        # Save the state_dict of the YOLO model
        torch.save(model.model.state_dict(), "yolo_weights.pth")

        import torch
        from ultralytics import YOLO

        # Instantiate the YOLO model
        model = YOLO()

        # Load the saved state_dict of the YOLO model
        model.model.load_state_dict(torch.load("yolo_weights.pth"))

        # Set the model to evaluation mode
        model.eval()

        return render(request,'objdec.html')