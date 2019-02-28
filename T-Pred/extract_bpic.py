import os

filename = "finale.csv"
outputfile = "helpdesk_cluster.txt"
outputfile2 = "helpdesk_extract.txt"

def extract_activity(filename, outputfile):
	readfile = open(filename, 'r')	
	writefile = open(outputfile, 'w+')
	lines = readfile.readlines()[2:]
	row_count = 0
	incident_id = ''
	for line in lines:
		line_list = line.split(",")
		if incident_id == line_list[0]:
			writefile.write(line_list[2].replace(" ",'').replace('-','') + "-" + line_list[3].replace(' ','') + "-" + line_list[4].replace(' ','') + line_list[5].replace(" ",'') + '-' + line_list[6].replace(" ",'').replace('\n','') + ' ')
		else:
			writefile.write("[EOC]-[EOC]-[EOC]-[EOC]\n")
			writefile.write(line_list[1].replace(' ','')+": ")
			writefile.write(line_list[2].replace(" ",'').replace('-','') + "-" + line_list[3].replace(' ','') + "-" + line_list[4].replace(' ','') + line_list[5].replace(" ",'') + '-' + line_list[6].replace(" ",'').replace('\n','') + ' ')
			row_count += 1
		incident_id = line_list[0]
	print(row_count)

	readfile.close()
	writefile.close()
	print("done!!!!!!!!!!")

def write_dict(file, the_dict):
	with open(file, 'w+') as f:
		for content in the_dict:
			f.write(content + ',' + the_dict[content] + '\n')

def extract(filename, outputfile):
	readfile = open(filename, 'r')	
	writefile = open(outputfile, 'w+')
	lines = readfile.readlines()[2:]
	row_count = 0
	incident_id = ''
	for line in lines:
		line_list = line.split(",")
		if incident_id == line_list[0]:
			writefile.write(line_list[2].replace(" ",'').replace('-','') + ' ')
		else:
			writefile.write("[EOC]\n")
			writefile.write(line_list[2].replace(" ",'').replace('-','') + ' ')
			row_count += 1
		incident_id = line_list[0]
	print(row_count)

	readfile.close()
	writefile.close()
	print("done!!!!!!!!!!")

#extract_activity(filename, outputfile)
extract(filename, outputfile2)