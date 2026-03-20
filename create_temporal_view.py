import os
import psycopg2
from dotenv import load_dotenv

load_dotenv("c:/cutoff-analysis-service/.env")

def create_temporal_view():
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    print("Dropping existing ml_features_v3 table if exists...")
    cur.execute("DROP TABLE IF EXISTS ml_features_v3;")
    
    print("Creating ml_features_v3 containing raw rounds joined with mapped historical metadata...")
    # LEFT JOIN raw_cutoffs with colleges based on string matching (college_name)
    query = """
    CREATE TABLE ml_features_v3 AS
    SELECT 
        rc.year,
        rc.round,
        rc.course_name,
        rc.specialization,
        rc.category,
        rc.quota,
        rc.exam_name,
        rc.gender,
        rc.institute_type,
        rc.college_name,
        rc.closing_rank,
        co.established_year,
        co.state,
        co.city,
        co.typeofuni,
        cr.academic AS rating_academic,
        cr.placement AS rating_placement,
        pl.avg_package
    FROM raw_cutoffs rc
    LEFT JOIN colleges co ON rc.college_name = co.college_name
    LEFT JOIN college_ratings cr ON co.college_id = cr.college_id AND rc.year = cr.year
    LEFT JOIN placements pl ON co.college_id = pl.college_id AND rc.year = pl.year
    WHERE rc.closing_rank IS NOT NULL AND rc.closing_rank > 0;
    """
    
    try:
        cur.execute(query)
        conn.commit()
        
        # Verify row count
        cur.execute("SELECT COUNT(*) FROM ml_features_v3;")
        count = cur.fetchone()[0]
        print(f"Successfully created ml_features_v3 with {count} records!")
        
    except Exception as e:
        print(f"Error creating temporal view: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    create_temporal_view()
