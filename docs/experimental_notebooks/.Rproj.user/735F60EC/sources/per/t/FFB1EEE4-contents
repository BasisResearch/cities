require(dagitty)

# with zones
zones_dag <- dagitty('dag {
    year [pos="0,2"]
    month [pos="1,2"]
    limit_con [pos="2,1"]
    parcel_area [pos="2,0"]
    ward_id [pos="3.5,0.2"]
    zone_id [pos = "1, 0"]
    neighborhood_id [pos = "4, 0"]
    housing_units [pos = "5,1"]
    past_reform [pos ="0, .5"]
    past_reform_by_zone [pos = "0,1"]
    
    
    year -> housing_units
    month -> housing_units
    limit_con -> housing_units
    parcel_area -> housing_units
    ward_id -> housing_units
    zone_id -> housing_units
    neighborhood_id -> housing_units
    
    neighborhood_id -> parcel_area
    zone_id -> past_reform_by_zone
    zone_id -> parcel_area
    past_reform -> past_reform_by_zone
    past_reform_by_zone -> limit_con
    
}')



plot(zones_dag)
paths(zones_dag,"limit_con","housing_units") 
adjustmentSets(zones_dag,"limit_con","housing_units",type = "all")
impliedConditionalIndependencies(zones_dag)




#---------------------------------------------------------------------------------------
# with distances
zones_dag <- dagitty('dag {
    year [pos="0,2"]
    month [pos="1,2"]
    limit_con [pos="2,1"]
    parcel_area [pos="2,0"]
    ward_id [pos="3.5,0.2"]
    zone_id [pos = "1, 0"]
    neighborhood_id [pos = "4, 0"]
    housing_units [pos = "5,1"]
    past_reform [pos ="0, .5"]
    past_reform_by_zone [pos = "0,1"]
    
    
    year -> housing_units
    month -> housing_units
    limit_con -> housing_units
    parcel_area -> housing_units
    ward_id -> housing_units
    zone_id -> housing_units
    neighborhood_id -> housing_units
    
    neighborhood_id -> parcel_area
    zone_id -> past_reform_by_zone
    zone_id -> parcel_area
    past_reform -> past_reform_by_zone
    past_reform_by_zone -> limit_con
    
}')



plot(zones_dag)
paths(zones_dag,"limit_con","housing_units") 
adjustmentSets(zones_dag,"limit_con","housing_units",type = "all")
impliedConditionalIndependencies(zones_dag)
