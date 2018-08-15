#!/usr/bin/python

import sys
import re


OwnerID, typeID, ansID,rowID1, rowID2= None, None, None,None,None

for line in sys.stdin:
    
    typeID=re.findall(r'(?<=PostTypeId=")(\d+)', line) #find the unique typeId in each line
    typeID=int(" ".join(typeID))
    
    #question line
    if typeID ==1:
        #find and output "AcceptedAnswerId" and "row Id" in question line 
        ansID= re.findall(r'(?<=AcceptedAnswerId=")(\d+)', line) 
        rowID1=re.findall(r'(?<=row Id=")(\d+)', line)

        
        if ansID:
            rowID1, ansID = int(rowID1[0]), int(ansID[0])
            print("{0}\t{1}".format(ansID, rowID1))
            
    #answer line
    if typeID==2:
        #find and output "OwnerUserId" and "row Id"(in relation to "AcceptedAnswerId") in answer line 
        rowID2=re.findall(r'(?<=row Id=")(\d+)', line)
        OwnerID=re.findall(r'(?<=OwnerUserId=")(\d+)', line)

        #r
        if OwnerID:
            rowID2, OwnerID = int(rowID2[0]), int(OwnerID[0])
            print("{0}\t'P'{1}".format(rowID2, OwnerID)) #add 'P' at the start of OwnerID in order to distinguish with question ID
        


    
   
