import random
from datetime import date

def load_data_pre():
    with open("./students.txt", "r", encoding="utf8") as file:
        students = file.readlines()
    students = "".join(students)
    students = students.split("\n")

    remove_list = input("빠지는 사람: ").split()
    if len(remove_list)!=0:
        for remove in remove_list:
            try: students.remove(remove)
            except: pass
    
    return students

def get_presenter():
    ymd = date.today().isoformat().split("-")
    students = load_data_pre()
    title = input("발표 주제: ").split()
    if len(title)==0:
        presenter = random.sample(students, 1)
        print(f"{ymd[0]}년 {ymd[1]}월 {ymd[2]}일 발표자: {presenter}")
    else:
        presenter = random.sample(students, len(title))
        print(f"{ymd[0]}년 {ymd[1]}월 {ymd[2]}일 주제별 발표자\n")
        print("*"*30, "\n")
        for idx, t, s in zip(range(len(title)), title, presenter):
            print(f"{idx+1}.{t}: [ {s} ]")
        print("\n", "*"*30, "\n")

get_presenter()