import xml.etree.ElementTree as  ET


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def output_ild(lane_nodes, secondary_lanes=None):
    output_str = '<additional>\n'
    ild = '  <laneAreaDetector file="ild.out" freq="1" id="%s" lane="%s" pos="-%.2f" endPos="-1"/>\n'
    for lane in lane_nodes:
        length = float(lane.attrib['length'])
        if length < 50.00:
            output_str += ild % (lane.attrib['id'], lane.attrib['id'], length)
        else:
            output_str += ild % (lane.attrib['id'], lane.attrib['id'], 50.00)
    if secondary_lanes is not None:
        for key, value in secondary_lanes.items():
            output_str += ild % (key, key, value)
    output_str += '</additional>\n'
    return output_str


if __name__ == '__main__':
    EXCEPTIONAL_LANES = ['233651288#1']

    tree = ET.parse("C:/Users/admin/Desktop/project/MLP_ATSC/seoul/data/seoul.net.xml")
    root = tree.getroot()

    #Find tl id
    tl_ids = []
    for tl in root.findall('tlLogic'):
        tl_ids.append(tl.attrib['id'])

    for tl in root.findall('tlLogic'):
        tl_ids.append(tl.attrib['id'])

    #Find all incoming lanes
    junctions = []
    incLanes = []
    for junction in root.iter('junction'):
        if junction.attrib['id'] in tl_ids:
            junctions.append(junction)
            for lane in junction.attrib['incLanes'].split(' '):
                incLanes.append(lane)
    lane_nodes = []
    short_lanes = dict()
    secondary_lanes = dict()
    secondary_lanes_map = dict()
    #Find all lane nodes
    for lane in root.iter('lane'):
        if lane.attrib['id'] in incLanes:
            lane_nodes.append(lane)
            length = float(lane.attrib['length'])
            if length < 50.00:
                short_lanes[lane.attrib['id']] = float(lane.attrib['length'])

    short_lanes_id = []
    for short_lane in short_lanes.keys():
        short_lanes_id.append(short_lane.split('_')[0])
  
    duplicated_connections = dict()
    for connection in root.iter('connection'):
        if  connection.attrib['to']  not in short_lanes_id or  connection.attrib['from'][0] == ':' or connection.attrib['from'] in EXCEPTIONAL_LANES:
            continue
        if connection.attrib['to'] + '_' + connection.attrib['toLane'] in short_lanes.keys():
            secondary_lanes[connection.attrib['from'] + '_' + connection.attrib['fromLane']] = 50.00 - short_lanes[connection.attrib['to'] + '_' + connection.attrib['toLane']]
            if connection.attrib['to'] + '_' + connection.attrib['toLane'] not in secondary_lanes_map:
                secondary_lanes_map[connection.attrib['to'] + '_' + connection.attrib['toLane']] = connection.attrib['from'] + '_' + connection.attrib['fromLane']
            else:
                print("Duplicated secondary lane: ", connection.attrib['to'] + '_' + connection.attrib['toLane'])
    
    print(secondary_lanes_map)
    #For seoul environment

    write_file('C:/Users/admin/Desktop/project/MLP_ATSC/seoul/data/seoul.add.xml', output_ild(lane_nodes, secondary_lanes))










     
        

    




