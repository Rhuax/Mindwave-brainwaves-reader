from NeuroPy import NeuroPy
import threading


def stoppa(a):
    a.stop()
    a.save()

#Rilassamento 438 secs
#Musica metal 200 secs
#Logica 200
#Memoria 200

person_name = input('Nome dell\'utente: ')
task_name = input('Nome del task: ')
task_duration = input('Durata del task(sec):')

record = NeuroPy("COM3", person_name=person_name, task_name=task_name, task_duration=task_duration)
threading.Timer(int(task_duration), stoppa, [record]).start()
record.start()
