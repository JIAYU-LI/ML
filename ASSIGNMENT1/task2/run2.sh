hadoop jar /opt/hadoop/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar -input /user/$USER/data/task1-output -output /user/$USER/data/task2-output -mapper mapper2.py -file mapper2.py -reducer reducer2.py -file reducer2.py
