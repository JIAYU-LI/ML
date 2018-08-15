hadoop jar /opt/hadoop/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar -D mapred.reduce.tasks=100 -input /user/$USER/data/task2-output -output /user/$USER/data/task4-output -mapper mapper4.py -file mapper4.py -reducer reducer4.py -file reducer4.py

