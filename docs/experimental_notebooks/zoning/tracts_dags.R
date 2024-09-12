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
paths(tracts_dag,"limit","units")
adjustmentSets(tracts_dag,"limit","units", type = "all")
impliedConditionalIndependencies(tracts_dag)
