import xml.etree.ElementTree as ET
import os
import sys
import argparse

def load_xml(path):
	print(path)
	lemma = os.path.basename(path).split('.xml')[0]
	tree = ET.parse(path)
	root = tree.getroot()
	role_set = []	# not a map!, we want to keep the original order as it appears because they are ordered by frequence
	for p in root.findall('predicate'):
		for roleset in p.findall('roleset'):
			frame_id = '.'.join(roleset.attrib['id'].split('.')[1:])
			arg_set = [get_arg_name(role) for role in roleset.find('roles').findall('role')]
			role_set.append((frame_id, arg_set))
	return lemma, role_set

def get_arg_name(node):
	if str.isdigit(node.attrib['n']):
		return node.attrib['n']
	else:
		return (node.attrib['n'] + "-" + node.attrib['f']).upper()

def extract(opt):
	cnt = 0
	with open(opt.output, 'w') as f:
		# the first entry is for empty
		f.write('{0}\t \n'.format('#'))
		cnt += 1

		for filename in os.listdir(opt.dir):
			if filename.endswith('.xml'):
				lemma, role_set = load_xml(opt.dir + '/' + filename)
				role_str = ' '.join(['{0}|{1}'.format(frame_id, ','.join(arg_set)) for frame_id, arg_set in role_set])
				f.write('{0} {1}\n'.format(lemma, role_str))
				cnt += 1
	print('{0} entries written to {1}'.format(cnt, opt.output))


def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dir', help="Path to frame xml files", default = "./data/propbank-frames/frames/")
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "./data/srl/frameset.txt")
	opt = parser.parse_args(arguments)

	extract(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
