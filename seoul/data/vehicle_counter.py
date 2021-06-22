import xml.etree.ElementTree as  ET

edge_map = {'517641017#2':'1_l', '220027347#2':'1_r', '184083187#2':'2_l', '782892488#1':'2_r', '218923156#5':'3_u',  '-218923156#4' : '3_d'}
count_map = {'1_l' : 0, '1_r' :0,'2_l' : 0, '2_r' :0, '3_u' : 0, '3_d' :0}

if __name__ == '__main__':
    flow_file  =  'seoul.rou.xml'
    tree = ET.parse(flow_file)
    root = tree.getroot()
    
    for vehicle in root.iter('vehicle'):
        for route in vehicle.iter('route'):
            for edge in route.attrib['edges'].split(' '):
                if edge in edge_map.keys():
                    count_map[edge_map[edge]] += 1
                    
    print(count_map)
                