import os
import numpy as np
import datetime
from tqdm import tqdm

''' 
for IPTV_data
'''

def parse_ymd(s):
    year_s, mon_s, day_s = s.split(' ')[0].split('/')
    hour_s, min_s, sec_s = s.split(' ')[1].split(':')
    return datetime.datetime(int(year_s), int(mon_s), int(day_s),
    	int(hour_s), int(min_s), int(sec_s))

path = "./T-pred-Dataset/IPTV_Data"
file_list = os.listdir(path)

event_file = 'IPTV_event.txt'
write_event = open(os.path.join(path, event_file), 'w+')
time_file = 'IPTV_time.txt'
write_time = open(os.path.join(path, time_file), 'w+')


for file_name in file_list:
	with open(os.path.join(path, file_name), 'r') as f:
		print(file_name)
		lines = f.readlines()[1:]
		current_time = lines[0].split(',')[1]
		current_time = parse_ymd(current_time)
		ID_list = []
		for i in range(len(lines)):
			ID, time, event = lines[i].replace('\n','').split(',')[0:3]
			ID_list.append(ID)
			time = parse_ymd(time)
			delta_time = (time - current_time).total_seconds()
			current_time = time
			write_event.write(event + '\t')
			write_time.write(str(delta_time) + '\t')
		if np.unique(np.array(ID_list)).shape[0] != 1:
			print('Multiple IDs: %s' % file_name)
		write_event.write('[EOS]\n')
		write_time.write('[EOS]\n')

'''
for music data
The timestamp is reverse order so the time dataset needs to be reversed after this extraction
'''
# def parse_ymd(s):
#     year_s, mon_s, day_s = s.split(' ')[0].split('-')
#     hour_s, min_s, sec_s = s.split(' ')[1].split(':')
#     return datetime.datetime(int(year_s), int(mon_s), int(day_s),
#     	int(hour_s), int(min_s), int(sec_s))

# path = "./T-pred-Dataset/lastfm-dataset-1K"

# file_name = 'userid-timestamp-artid-artname-traid-traname.tsv'
# event_file = 'lastfm_1k_event.txt'
# write_event = open(os.path.join(path, event_file), 'w+', encoding='UTF-8')
# time_file = 'lastfm_1k_time.txt'
# write_time = open(os.path.join(path, time_file), 'w+', encoding='UTF-8')


# with open(os.path.join(path, file_name), 'r', encoding='UTF-8') as f:
# 	print(file_name)
# 	lines = f.readlines()
# 	current_time = lines[0].split('\t')[1].replace('T',' ').replace('Z','')
# 	print(current_time)
# 	current_time = parse_ymd(current_time)
# 	current_id = lines[0].split('\t')[0]
# 	ID_list = []
# 	for i in tqdm(range(len(lines))):
# 		ID, time = lines[i].split('\t')[0:2]
# 		event = lines[i].replace('\n', '').split(',')[-1]
# 		ID_list.append(ID)
# 		time = parse_ymd(time.replace('T',' ').replace('Z',''))
# 		if ID == current_id:
# 			delta_time = (time - current_time).total_seconds()
# 			current_time = time
# 			write_event.write(event + '\t')
# 			write_time.write(str(delta_time) + '\t')
# 			current_id = ID
# 		else:
# 			write_event.write('[EOS]\n')
# 			write_time.write('[EOS]\n')
# 			delta_time = 0.0
# 			current_time = time
# 			write_event.write(event + '\t')
# 			write_time.write(str(delta_time) + '\t')
# 			current_id = ID

# 	write_event.write('[EOS]\n')
# 	write_time.write('[EOS]\n')

# 	print('Multiple IDs: %d' % np.unique(np.array(ID_list)).shape[0])

'''for stack overflow data'''

# def parse_ymd(s):
#     year_s, mon_s, day_s = s.split(' ')[0].split('/')
#     hour_s, min_s, sec_s = s.split(' ')[1].split(':')
#     return datetime.datetime(int(year_s), int(mon_s), int(day_s),
#     	int(hour_s), int(min_s), int(sec_s))

# path = "./T-pred-Dataset/StackOverflowData"
# file_list = os.listdir(path)
# file_name = "answers.csv"

# event_file = 'stackof_event.txt'
# write_event = open(os.path.join(path, event_file), 'w+')
# time_file = 'stackof_time.txt'
# write_time = open(os.path.join(path, time_file), 'w+')

# with open(os.path.join(path, file_name), 'r', encoding='UTF-8') as f:
# 	print(file_name)
# 	lines = f.readlines()
# 	print(lines[0])
# 	lines = lines[1:]
# 	current_time = lines[0].split(',')[-1]
# 	current_time = int(current_time)
# 	current_id = lines[0].split(',')[1]
# 	ID_list = []
# 	for i in tqdm(range(len(lines))):
# 		ID = lines[i].split(',')[1]
# 		time = int(lines[i].split(',')[-1])
# 		event = lines[i].split(',')[-2]
# 		# print('%s, %d, %s' %(ID, time, event))
# 		ID_list.append(ID)
# 		if ID == current_id:
# 			delta_time = time - current_time
# 			current_time = time
# 			write_event.write(event + '\t')
# 			write_time.write(str(delta_time) + '\t')
# 			current_id = ID
# 		else:
# 			write_event.write('[EOS]\n')
# 			write_time.write('[EOS]\n')
# 			delta_time = 0
# 			current_time = time
# 			write_event.write(event + '\t')
# 			write_time.write(str(delta_time) + '\t')
# 			current_id = ID

# 	write_event.write('[EOS]\n')
# 	write_time.write('[EOS]\n')

# 	print('Multiple IDs: %d' % np.unique(np.array(ID_list)).shape[0])
