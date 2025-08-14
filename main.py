from NeuroPy import NeuroPy
import threading


def stoppa(a):
    a.stop()
    a.save()

#Rilassamento 438 secs a 1min
#Musica metal 200 secs da 25sec
#Logica 200 da 0
#Memoria 200 da 0

person_name = "a"#input('Nome dell\'utente: ')
task_name =  "a"#input('Nome del task: ')
task_duration =  "20"#input('Durata del task(sec):')

record = NeuroPy("COM3", person_name=person_name, task_name=task_name, task_duration=task_duration)
threading.Timer(int(task_duration), stoppa, [record]).start()
record.start()
