import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
cred = credentials.Certificate("/Users/rishirajrajawat/PycharmProjects/Veriface123/.venv/veriface-9c99e-firebase-adminsdk-lkefp-a54bf03d00.json")
firebase_admin.initialize_app(cred,{
    "databaseURL": "https://veriface-9c99e-default-rtdb.firebaseio.com/",
    "storageBucket" : "gs://veriface-9c99e.appspot.com"
})

ref = db.reference('Student')


data = {
    "510555":
        {
            "name": "RISHIRAJ SINGH RAJAWAT",
            "major": "CSDA",
            "starting_year":2023,
            "total_attendance":6,
            "standing": "A",
            "year":1,
            "last_attendance_time": "2024-12-11 00:54:34"

        },
    "202020":
        {
            "name": "hhhhhh",
            "major": "CSDA",
            "starting_year": 2021,
            "total_attendance": 12,
            "standing": "B",
            "year": 1,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "303030":
        {
            "name": "REHAN",
            "major": "CSDA",
            "starting_year": 2020,
            "total_attendance": 7,
            "standing": "G",
            "year": 2,
            "last_attendance_time": "2022-12-11 00:54:34"
        }



}

for key, value in data.items():
    ref.child(key).set(value)