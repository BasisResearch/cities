require(dagitty)

# with zones
tracts_dag <- dagitty('dag {
    year [pos="0,2"]
    distance [pos = "0,0"]

    white [pos = ".2,1"]
    segregation [pos = ".6,1"]
    
    income [pos = ".9,.8"]

    median_value [pos = "1.2,0.2"]
    limit [pos=".7,1.8"]
    units [pos = "1.5,.8"]
    
    sqm [pos = ".2,.4"]
   
    distance -> sqm
    year -> sqm
   
    year -> limit
    distance -> limit
   
    distance -> white
    year -> white
    sqm -> white
    limit -> white
    
    sqm -> segregation
    distance -> segregation
    white -> segregation
    year -> segregation
    limit -> segregation
    
    
    
    sqm -> income
    distance -> income
    white -> income
    segregation -> income
    year -> income
    limit -> income
    
 
    
    sqm -> median_value
    distance -> median_value
    limit -> median_value
    income -> median_value
    white -> median_value
    segregation -> median_value
    year -> median_value
    
  
    
    sqm -> units
    median_value -> units
    distance -> units
    income -> units
    white -> units
    limit -> units
    segregation -> units
    year -> units
    }')



plot(tracts_dag)
paths(tracts_dag,"limit","units")
adjustmentSets(tracts_dag,"limit","units", type = "all")
impliedConditionalIndependencies(tracts_dag)
