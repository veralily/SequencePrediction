import os
import datetime

def parse_ymd(s):
	year_s, mon_s, day_s = s.split('T')[0].split('-')
	hour_s, min_s, sec_s = s.split('T')[1].split('.')[0].split(':')
	return datetime.datetime(int(year_s), int(mon_s), int(day_s), int(hour_s), int(min_s), int(sec_s))

filename = "yoochoose-clicks.txt"

output_event = "RECSYS15_event.txt"
output_time = "RECSYS15_time.txt"
eventfile = open(output_event, 'w+')
timefile = open(output_time, 'w+')

with open(filename, 'r') as f:
	lines = f.readlines()
	currentID = lines[0].split(',')[0]
	current_time = lines[0].split(',')[1]
	current_time = parse_ymd(current_time)
	for line in lines:
		line = line.split(',')
		sessID = line[0]
		time = parse_ymd(line[1])
		event = line[2]		
		if sessID == currentID:
			delta_time = (time - current_time).total_seconds()
			timefile.write(str(delta_time) + '\t')
			eventfile.write(event + '\t')
		else:
			timefile.write('[EOS]\n')
			eventfile.write('[EOS]\n')
			timefile.write('0.0\t')
			eventfile.write(event + '\t')
		currentID = sessID
		current_time = time
	f.close()

timefile.close()
eventfile.close()