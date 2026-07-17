import ee


PROJECT_ID = "geoai-thaton-flood"


def main():
    try:
        ee.Initialize(project=PROJECT_ID)

        print("Earth Engine already authenticated.")
        print("Project:", PROJECT_ID)

    except Exception:
        print("Earth Engine authentication required.")

        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

        print("Earth Engine authentication successful.")
        print("Project:", PROJECT_ID)

    # Test GPM access
    gpm = ee.ImageCollection(
        "NASA/GPM_L3/IMERG_V07"
    )

    print(
        "GPM image count:",
        gpm.size().getInfo(),
    )


if __name__ == "__main__":
    main()