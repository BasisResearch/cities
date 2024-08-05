zip_load = """
select zcta5ce20, 2020, geom from zip_raw_2020
union select zcta, 2000, geom from zip_raw_2000
"""
print("Executing:", zip_load)
cur.execute(zip_load)
conn.commit()
