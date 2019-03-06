import xml.etree.ElementTree
import glob
import os
import json

def getSize(file):
    size = str(os.stat(file).st_size)

    return size

def swapExtension(file):
    name = str.split(str(file), ".")[0]
    ext = ".jpg"
    
    return name+ext
# f= open("guru99.txt","w+")
    # ormat = {"2012-09-19_06_00_47.jpg224714":{"filename":"2012-09-19_06_00_47.jpg","size":224714,"regions":[{"shape_attributes":{"name":"polyline","all_points_x":[403,420,430,414,365,369,381,401],"all_points_y":[417,424,438,473,474,443,421,416]},"region_attributes":{"Type":"Occupied"}},{"shape_attributes":{"name":"polyline","all_points_x":[501,490,547,557,502],"all_points_y":[423,475,475,422,424]},"region_attributes":{"Type":"Empty"}}],"file_attributes":{}}}
def pklot(Directory):
    count = 0
    car = ""
    # file = "2012-09-19_05_55_47"
    os.chdir(Directory)
    shape_attributes = " "
    for file in glob.glob("*.xml"):
        filename = swapExtension(file)
        # Each box will need its own regionData members New one every loop
        # regionData = "shape_attributes:",
        
        
        
        filestart='{"'+filename + getSize(file)+'":'
        
        # size = os.stat(file).st_size
        root = xml.etree.ElementTree.parse(file).getroot()
        for space in root.findall('space'):
            x_points=""
            y_points=""
            occupied = space.get('occupied')
            if occupied == "0":
                car = "Empty"
            else:
                car = "Occupied"
            for i in range(0, 4):
                x = space.find('contour')[i].get('x')
                
                x_points += x 
                if i != 3:
                    x_points += ","
                y = space.find('contour')[i].get('y')
                y_points += y 
                if i != 3:
                    y_points += ","
            count += 1
            
            if file == glob.glob("*.xml")[-1] and space == root.findall('space')[-1]:
                new_attributes='{"' + 'shape_attributes' + '":' + '{"name":"polyline", "all_points_x"' + ':[' + x_points + '],' + '"all_points_y"' + ':[' + y_points + ']},'  +  '"region_attributes":{"Type"' + ':"' + car + '"}' + '}'

            else:
                new_attributes='{"' + 'shape_attributes' + '":' + '{"name":"polyline", "all_points_x"' + ':[' + x_points + '],' + '"all_points_y"' + ':[' + y_points + ']},'  +  '"region_attributes":{"Type"' + ':"' + car + '"}' + '},'
            shape_attributes += str(new_attributes)
    data = '{"' + 'filename' + '":"' + filename + '","' + 'size' + '":"'  + getSize(file) + '","' +   'regions' + '":[' + shape_attributes + '],"file_attributes"' + ':' + '{'+'}' + '}'+ '}'
    # print(filestart + str(data))
    f= open("via_region_data.json","w+")
    f.write(filestart + data)
    f.close()


def countDataPoints(Directory):
        count = 0
        occupied0 = 0
        occupied1 = 0
        os.chdir(Directory)
        for file in glob.glob("*.xml"):
            # print(file)
            root = xml.etree.ElementTree.parse(file).getroot()
            for space in root.findall('space'):
                occupied = space.get('occupied')
                if occupied == "0":
                    occupied0 += 1
                else:
                    occupied1 += 1
                for i in range(0, 4):
                    x = space.find('contour')[i].get('x')
                    y = space.find('contour')[i].get('y')
                    # print(occupied, x, y)
                count += 1
        print(count,"Spaces", occupied0,"Empty", occupied1, "Occupied" )


def countAnnotationPoints(Directory):
        count = 0
        occupied0 = 0
        occupied1 = 0
        os.chdir(Directory)
        for file in glob.glob("*.xml"):
            # print(file)
            root = xml.etree.ElementTree.parse(file).getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                print(name)
                for i in range(0, len(obj.find('polygon').find('pt'))):
                    x = obj.find('polygon')[i].find('pt').find('x').text
                    y = obj.find('polygon')[i].find('pt').find('y').text
                    print("Car at: ", x, y)
                count += 1
        print(count,"Spaces", occupied0,"Empty", occupied1, "Occupied" )
                    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument("command",
                        metavar="<Functioin>")
    parser.add_argument("directory",
                        metavar="<Directory>")
    args = parser.parse_args()

    # Train or evaluate
    if args.command == "pklot":
        pklot(args.directory)
    elif args.command == "count":
        countDataPoints(args.directory)
    elif args.command == "countAnno":
        countAnnotationPoints(args.directory)
    else:
        print("'{}' is not recognized. "
              "Use 'pklot' or 'count'".format(args.command))
