import multiprocessing, time, signal
p = multiprocessing.Process(target=time.sleep, args=(1000,))
p.start()
time.sleep(5)
if p.is_alive() is True:
	p.terminate()
print p, p.is_alive()

if __name__ == '__main__':
	freeze_support()