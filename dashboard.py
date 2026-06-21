def dashboard_data(results):

    total=len(results)

    selected=0

    for r in results:

        if r["result"]=="Selected":

            selected+=1


    rejected=total-selected


    return {

        "total":total,
        "selected":selected,
        "rejected":rejected

    }