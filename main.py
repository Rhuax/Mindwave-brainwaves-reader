from NeuroPy import NeuroPy
import threading


def stoppa(a):
    a.stop()
    a.save()


person_name = input('Nome dell\'utente: ')
task_name = input('Nome del task: ')
task_duration = input('Durata del task(sec):')

record = NeuroPy("COM3", person_name=person_name, task_name=task_name, task_duration=task_duration)
threading.Timer(task_duration, stoppa, [record]).start()
record.start()
