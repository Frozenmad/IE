#Examine the train and test file's format
import codecs
import yaml

def check(path,name):
	print("Examining the "+name+' file......')
	file = codecs.open(path, 'r', encoding='utf-8')
	line = file.readline()
	line_num = 1
	alert = False
	while line:
		line = line.rstrip()
		if not line:
			line = file.readline().rstrip()
			line_num += 1
			if line:
				print("Alert! Line {} should be empty!".format(line_num))
				alert = True
			line = file.readline()
			line_num += 1
			continue
		label = line.split('\t')
		if(len(label)!=3):
			print("Alert! Line {} has less than 3 labels!".format(line_num))
		line = file.readline()
		line_num += 1
	if not alert:
		print("No risk in " + name + " file!")
	file.close()


def main():
    with open('./config.yml',encoding="utf-8") as file_config:
        config = yaml.load(file_config)

        path_train = config['data_params']['path_train']
        path_test = config['data_params']['path_test']

        check(path_train,'train')
        check(path_test,'test')

if __name__ == '__main__':
	main()