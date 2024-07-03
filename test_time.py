import datetime
ts = 1535385100398781
ts_mb = 16890516615

#2018-08-27 23:51:40.398781
timestamp = ts / 1000000
timestamp_mb = ts_mb / 10

dt_object = datetime.datetime.fromtimestamp(ts)

print(dt_object)

