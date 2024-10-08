require(dagitty)

if (rstudioapi::isAvailable()) {
  # Set working directory to the script's location
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

print(getwd())

# with zones
tracts_dag <- dagitty('dag {
    year [pos="0,2"]
    distance [pos = "0,0"]
    
    square_meters [pos = "0.2,.4"]
    limit [pos = "0.2, 1.6"]
    
    white [pos = "0.4,1.8"]
    segregation [pos = "0.6,0.2"]
    
    income [pos = "0.7, .8"]
    
    median_value [pos = "0.9,1.4"]
    housing_units [pos = "1.,.6"]
    
    distance -> square_meters
    year -> square_meters
    
    distance -> limit
    year -> limit
    
    distance -> white
    square_meters -> white
    limit -> white
    
    distance -> segregation
    year -> segregation
    limit -> segregation
    square_meters -> segregation
    white -> segregation
    
    distance -> income
    white -> income
    segregation -> income
    square_meters -> income
    limit -> income
    year -> income
    
    distance -> median_value
    income  -> median_value
    white -> median_value
    segregation -> median_value
    square_meters -> median_value
    limit -> median_value
    year -> median_value
    
    median_value -> housing_units
    distance -> housing_units
    income -> housing_units
    white -> housing_units
    limit -> housing_units
    segregation -> housing_units
    square_meters -> housing_units
    year -> housing_units
    
    
    
    
    }')

plot(tracts_dag)


png("tracts_dag_plot_high_density.png",
    width = 2000,       
    height = 1600,     
    res = 300          
)
plot(tracts_dag)
dev.off()

pdf("tracts_dag_plot.pdf", 
    width = 10,        
    height = 8,        
    pointsize = 18,    
    paper = "special",
    useDingbats = FALSE, 
    compress = FALSE)    
plot(tracts_dag)
dev.off()

plot(tracts_dag)
paths(tracts_dag,"limit","housing_units")
adjustmentSets(tracts_dag,"limit","housing_units", type = "all")
impliedConditionalIndependencies(tracts_dag)
