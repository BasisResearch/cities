require(dagitty)

# with zones
tracts_dag <- dagitty('dag {
    year [pos="0,2"]
    distance [pos = "0,0"]
    total_value [pos = "1,0"]
    median_value [pos = "1.2,0.3"]
    limit [pos="1,1"]
    units [pos = "2,1"]
    
    distance -> limit
    distance -> total_value
    distance -> median_value
    distance -> units
    
    year -> limit
    year -> total_value
    year -> median_value
    year -> units
    
    total_value -> units
    median_value -> units
    
    
    limit -> total_value
    limit -> median_value
    limit -> units
    
    }')



plot(tracts_dag)
paths(tracts_dag,"limit","units")
adjustmentSets(tracts_dag,"limit","units", type = "all")
impliedConditionalIndependencies(tracts_dag)
