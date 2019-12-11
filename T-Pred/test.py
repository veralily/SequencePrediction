import read_data

event_file = './T-pred-Dataset/lastfm-v5k_event.txt'
time_file = './T-pred-Dataset/lastfm-v5k_time2.txt'

train_data, valid_data, test_data = read_data.data_split(event_file, time_file, shuffle=True)
input_event_data, target_event_data, input_time_data, target_time_data = read_data.data_iterator(valid_data,20,5)
# a,b,c,d = read_data.generate_batch(100,input_event_data,target_event_data,input_time_data,target_time_data)

