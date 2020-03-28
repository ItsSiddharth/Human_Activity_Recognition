import subprocess
duration = subprocess.check_output(['ffprobe', '-i', 'test_video.mp4', '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")])
print(int(float((str(duration)[2:-3]))))


