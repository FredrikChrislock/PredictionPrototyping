
InterestingIntervals = function(TurbineA, TurbineB, IntTime, Margin) {
    # Returns important periods where Turbine A and Turbine B have worked for more than IntTime hours,
    # before either Turbine A, Turbine B or both trips. In addition the function returns the periods
    # where the system is expected to be in the normal case.
    library(lubridate)
    library(base)
    # IntTime pick time intervals in which we are interested in
    # for further investigations with respect to when TurbineA and TurbineB trips.
    # Margin gives the time period in which a normal period must be after a start up and before a shutdown (in days). 

    # MarginTimeBeforeShutdown is wanted interested time before shutdown.
    # If a tripping of either turbine has happened within MarginTimeBeforeShutdown,
    # the interested data will then be BEFORE THAT TRIPPING.


    # DEF: INTERESTING POINTS
    # --------------------------------------------------------------------------
    # Interesting points here are regions just before trippings of either Turbin A or Turbine B or both)
    # The important thing is that BOTH works before the tripping, and they work in a LARGE ENOUGH period
    # given by a SELF-SELECTED time period given by the input variable(IMPORTANT!): IntTime (in hours)
    # and normal behaviour.
    # Turbine A contains data of wheather turbine A is working or not. 
    # Turbine B - .. -
    # --------------------------------------------------------------------------

    # The data frame of TurbineA/TurbineB is: column 1: tag name, column 2: data value, column 3: date, column 4: time

    # IMPORTANT: The function assumes no NA-variables
    #-------------------------------------------------------------------------------------------------------------

    # FIRST, DETECT THE TIME POINTS OF WHICH TRIPPINGS AND START-UPS HAPPENS FOR TURBINE A AND B

    TrippingsA = c()
    TrippingsB = c()
    # Vector showing where turbine A and B trips.
    TurbineAStartingAGain = c()
    TurbineBStartingAGain = c()
    # Vector showing where turbine A and B starts up again after tripping period. 

    for (i in 2:nrow(TurbineA)) {

        if (TurbineA[i, 2] == 0 && TurbineA[i - 1, 2] == 1) {
            TrippingsA = c(TrippingsA, i)
        }

        if (TurbineA[i, 2] == 1 && TurbineA[i - 1, 2] == 0) {
            TurbineAStartingAGain = c(TurbineAStartingAGain, i)
        }

        if (TurbineB[i, 2] == 0 && TurbineB[i - 1, 2] == 1) {
            TrippingsB = c(TrippingsB, i)
        }

        if (TurbineB[i, 2] == 1 && TurbineB[i - 1, 2] == 0) {
            TurbineBStartingAGain = c(TurbineBStartingAGain, i)
        }

    }
    # -----------------------------------------------------------------------------------------------
    # NOW WE WANT TO EXTRACT PERIODS WHERE TURBINE A AND TURBINE B ARE WORKING INDEPENDENTLY:
    TimePeriodsWhereTurbineAWorks = list()
    # TURBINE A:
    # IF WE SEE A TRIPPING FIRST IN THE DATA:
    if (TrippingsA[1] < TurbineAStartingAGain[1]) {
        TimePeriodsWhereTurbineAWorks[[1]] = c(1, TrippingsA[1])

        w = 1
        while (length(TrippingsA) >= w + 1) {
            TimePeriodsWhereTurbineAWorks[[w + 1]] = c(TurbineAStartingAGain[w], TrippingsA[w + 1])
            w = w + 1
        }
    }

    # ELSE, WE SEE A START-UP FIRST IN THE DATA:
    else {
        w = 1
        while (w <= length(TrippingsA)) {
            TimePeriodsWhereTurbineAWorks[[w]] = c(TurbineAStartingAGain[w], TrippingsA[w])
        }
    }

    # TURBINE B:
    TimePeriodsWhereTurbineBWorks = list()
    # IF WE SEE B TRIPPING FIRST IN THE DATA:
    if (TrippingsB[1] < TurbineBStartingAGain[1]) {
        TimePeriodsWhereTurbineBWorks[[1]] = c(1, TrippingsB[1])

        w = 1
        while (length(TrippingsB) >= w + 1) {
            TimePeriodsWhereTurbineBWorks[[w + 1]] = c(TurbineBStartingAGain[w], TrippingsB[w + 1])
            w = w + 1
        }
    }

    # ELSE, WE SEE A START-UP FIRST IN THE DATA:
    else {
        w = 1
        while (w <= length(TrippingsB)) {
            TimePeriodsWhereTurbineBWorks[[w]] = c(TurbineBStartingAGain[w], TrippingsB[w])
        }
    }
    # ---------------------------------------------------------------------------------------------------
    # NOW WE WANT TO TAKE OUT REALLY INTERESTING TIME INTERVALS! (SEE DEFINITION OF INTERESTING TIME INTERVALS ABOVE)

    # TURBINE A:
    # Chech if the given time intervals where turbine A is working is long enough, in other words longer than intTime:
    InterestingTimesForTurbineA = list()
    for (i in 1:length(TimePeriodsWhereTurbineAWorks)) {
        # Start up time:
        start = paste(TurbineA[TimePeriodsWhereTurbineAWorks[[i]][1], 3], TurbineA[TimePeriodsWhereTurbineAWorks[[i]][1], 4], sep = " ")
        # Tripping time:
        end = paste(TurbineA[TimePeriodsWhereTurbineAWorks[[i]][2], 3], TurbineA[TimePeriodsWhereTurbineAWorks[[i]][2], 4], sep = " ")

        # Chech if interval is greater than IntTime
        diff = as.numeric(difftime(end, start, units = "hours"))
        if (diff >= IntTime) {
            InterestingTimesForTurbineA[[length(InterestingTimesForTurbineA)+1]] = TimePeriodsWhereTurbineAWorks[[i]]
        }
    }

    # TURBINE B:
    # Chech if the given time intervals where turbine B is working is long enough, in other words longer than intTime:
    InterestingTimesForTurbineB = list()
    for (i in 1:length(TimePeriodsWhereTurbineBWorks)) {
        # Start up time:
        start = paste(TurbineB[TimePeriodsWhereTurbineBWorks[[i]][1], 3], TurbineB[TimePeriodsWhereTurbineBWorks[[i]][1], 4], sep = " ")
        # Tripping time:
        end = paste(TurbineB[TimePeriodsWhereTurbineBWorks[[i]][2], 3], TurbineB[TimePeriodsWhereTurbineBWorks[[i]][2], 4], sep = " ")

        # Chech if interval is greater than IntTime
        diff = as.numeric(difftime(end, start, units = "hours"))
        if (diff >= IntTime) {
            InterestingTimesForTurbineB[[length(InterestingTimesForTurbineB) + 1]] = TimePeriodsWhereTurbineBWorks[[i]]
        }
    }
#---------------------------------------------------------------------------------------------------------------------------
    # NOW WE WANT TO CHECH THAT TURBINE B ALSO WORKS FOR MORE THAN IntTime HOURS BEFORE TURBINE A TRIPS. (VERY IMPORTANT STEP!)
    # NB! Code can be improved to optimize speed. This is a dummy code!
    AnalysisTimesForA = list()
    for (i in 1:length(InterestingTimesForTurbineA)) {
        #Chech that turbine B also worked just before turbine A tripped.
        JustBeforeTrip = paste(TurbineA[InterestingTimesForTurbineA[[i]][2] - 1, 3], TurbineA[InterestingTimesForTurbineA[[i]][2] - 1, 4], sep = " ")
        #Chech if the time point is within any interesting time interval of B, and if this is the case, see if turbine B has worked for longer than IntTime:
        for (j in 1:length(InterestingTimesForTurbineB)) {

            diff1 = as.numeric(difftime(JustBeforeTrip, paste(TurbineB[InterestingTimesForTurbineB[[j]][1], 3], TurbineB[InterestingTimesForTurbineB[[j]][1], 4], sep = " "),units = "hours"))
            diff2 = as.numeric(difftime(paste(TurbineB[InterestingTimesForTurbineB[[j]][2]-1, 3], TurbineB[InterestingTimesForTurbineB[[j]][2]-1, 4], sep = " "), JustBeforeTrip, units = "hours"))
            # IF JustBeforeTrip is within one time interval where Turbine B works more than IntTime:
            #Chech for how long Turbine B has worked before tripping of turbine A.
            if (diff1 >= IntTime && diff2 >= 0) {

                # Find out which of turbine A and B that started first:
                v = as.numeric(difftime(paste(TurbineB[InterestingTimesForTurbineB[[j]][1], 3], TurbineB[InterestingTimesForTurbineB[[j]][1], 4], sep = " "), paste(TurbineA[InterestingTimesForTurbineA[[i]][1], 3], TurbineA[InterestingTimesForTurbineA[[i]][1], 4], sep = " ")))
                
                if (v > 0) {
                    AnalysisTimesForA[[length(AnalysisTimesForA) + 1]] = c(paste(TurbineB[InterestingTimesForTurbineB[[j]][1], 3], TurbineB[InterestingTimesForTurbineB[[j]][1], 4], sep = " "), JustBeforeTrip)
                    break
                    # We break the inner for-loop because JustBeforeTrip contained in the time interval of A can be within only ONE interesting time interval for turbine B.
                }
                else {
                    AnalysisTimesForA[[length(AnalysisTimesForA) + 1]] = c(paste(TurbineA[InterestingTimesForTurbineA[[i]][1], 3], TurbineA[InterestingTimesForTurbineA[[i]][1], 4], sep = " "), JustBeforeTrip)
                    break
                }
               
            }

        }
    }
    # CHECH THE SAME FOR TURBINE B:
    AnalysisTimesForB = list()
    for (i in 1:length(InterestingTimesForTurbineB)) {
        #Chech that turbine A also worked just before turbine B tripped.
        JustBeforeTrip = paste(TurbineB[InterestingTimesForTurbineB[[i]][2] - 1, 3], TurbineB[InterestingTimesForTurbineB[[i]][2] - 1, 4], sep = " ")
        #Chech if the time point is within any interesting time interval of A, and if this is the case, see if turbine A has worked for longer than IntTime:
        for (j in 1:length(InterestingTimesForTurbineA)) {

            diff1 = as.numeric(difftime(JustBeforeTrip, paste(TurbineA[InterestingTimesForTurbineA[[j]][1], 3], TurbineA[InterestingTimesForTurbineA[[j]][1], 4], sep = " "), units = "hours"))
            diff2 = as.numeric(difftime(paste(TurbineA[InterestingTimesForTurbineA[[j]][2] - 1, 3], TurbineA[InterestingTimesForTurbineA[[j]][2] - 1, 4], sep = " "), JustBeforeTrip, units = "hours"))
            # IF JustBeforeTrip is within one time interval where Turbine B works more than IntTime:
            #Chech for how long Turbine B has worked before tripping of turbine A.
            if (diff1 >= IntTime && diff2 >= 0) {

                # Find out which of turbine A and B that started first:
                v = as.numeric(difftime(paste(TurbineA[InterestingTimesForTurbineA[[j]][1], 3], TurbineA[InterestingTimesForTurbineA[[j]][1], 4], sep = " "), paste(TurbineB[InterestingTimesForTurbineB[[i]][1], 3], TurbineB[InterestingTimesForTurbineB[[i]][1], 4], sep = " ")))

                if (v > 0) {
                    AnalysisTimesForB[[length(AnalysisTimesForB) + 1]] = c(paste(TurbineA[InterestingTimesForTurbineA[[j]][1], 3], TurbineA[InterestingTimesForTurbineA[[j]][1], 4], sep = " "), JustBeforeTrip)
                    break
                    # We break the inner for-loop because JustBeforeTrip contained in the time interval of B can be within only ONE interesting time interval for turbine A.
                }
                else {
                    AnalysisTimesForB[[length(AnalysisTimesForB) + 1]] = c(paste(TurbineB[InterestingTimesForTurbineB[[i]][1], 3], TurbineB[InterestingTimesForTurbineB[[i]][1], 4], sep = " "), JustBeforeTrip)
                    break
                }

            }

        }
    }
# ----------------------------------------------------------------------------------------------------------------------------
    # Return the important periods for both turbines + normal mode:

    # IntTime BEFORE TRIPS:
    matrixA = matrix(nrow = length(AnalysisTimesForA), ncol = 2)
    matrixB = matrix(nrow = length(AnalysisTimesForB), ncol = 2)
    
    for (i in 1:length(AnalysisTimesForA)) {
        matrixA[i,] = AnalysisTimesForA[[i]]
    }

    for (i in 1:length(AnalysisTimesForB)) {
        matrixB[i,] = AnalysisTimesForB[[i]]
    }

    # NORMAL MODE:
    # Yet again, this is a dummy code. Code can be improved:
    # Find the largest period in which both turbine A and turbine B works:

    TimeMaxA = 0
    MaxRangeA = 0
    for (i in 1:nrow(matrixA)) {
        rangeA = as.numeric(difftime(matrixA[i, 2], matrixA[i, 1]), units = "hours")
        if (rangeA > MaxRangeA) {
            TimeMaxA = matrixA[i,]
            MaxRangeA  = rangeA
        }
    }
    TimeMaxB = 0
    MaxRangeB = 0
    for (i in 1:nrow(matrixB)) {
        rangeB = as.numeric(difftime(matrixB[i, 2], matrixB[i, 1]), units = "hours")
        if (rangeB > MaxRangeB) {
            TimeMaxB = matrixB[i,]
            MaxRangeB = rangeB
        }
    }
    
    NormalTime = c()
    if (MaxRangeA >= MaxRangeB) {
        NormalTime[1] = as.character(as.POSIXct((TimeMaxA[1])) + days(Margin))
        NormalTime[2] = as.character(as.POSIXct((TimeMaxA[2])) - days(Margin))
    }
    else {
        NormalTime[1] = as.character(as.POSIXct(TimeMaxB[1]) + days(Margin))
        NormalTime[2] = as.character(as.POSIXct(TimeMaxB[2]) - days(Margin))
    }


    #colnames(matrixA) = c("Turbine A intTime before trip", "Turbine A just before trip")
    #colnames(matrixB) = c("Turbine B intTime before trip", "Turbine B just before trip")

    # Returns important times for turbine A and turbine B before trips where both A and B have worked for more than
    # intTime hours before trip. In addition a period of normal behaviour is returned. 
    l = list(matrixA, matrixB, NormalTime)

    return(l)


    }