import os
import psycopg2
from dotenv import load_dotenv

load_dotenv("c:/cutoff-analysis-service/.env")

def create_ml_view():
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    print("Dropping existing ml_features_v2 table if exists...")
    cur.execute("DROP TABLE IF EXISTS ml_features_v2;")
    
    print("Creating and populating ml_features_v2 with joined historical data...")
    # Constructing the massive JOIN query
    query = """
    CREATE TABLE ml_features_v2 AS
    SELECT 
        c.cutoff_value,
        c.year,
        c.category,
        c.cutoff_type,
        co.college_name,
        co.established_year,
        co.state,
        co.city,
        co.typeofuni,
        crs.course_name,
        sp.specialization_name,
        ex.name AS exam_name,
        cr.hostel AS rating_hostel,
        cr.academic AS rating_academic,
        cr.faculty AS rating_faculty,
        cr.infra AS rating_infra,
        cr.placement AS rating_placement,
        pl.highest_package,
        pl.avg_package,
        cs.fees,
        cs.duration_years
    FROM cutoffs c
    LEFT JOIN colleges co ON c.college_id = co.college_id
    LEFT JOIN courses crs ON c.course_id = crs.course_id
    LEFT JOIN specializations sp ON c.specialization_id = sp.specialization_id
    LEFT JOIN exams ex ON c.exam_id = ex.exam_id
    LEFT JOIN college_ratings cr ON co.college_id = cr.college_id AND c.year = cr.year
    LEFT JOIN placements pl ON co.college_id = pl.college_id AND c.year = pl.year
    LEFT JOIN college_specializations cs ON co.college_id = cs.college_id AND c.course_id = cs.course_id AND c.specialization_id = cs.specialization_id;
    """
    
    try:
        cur.execute(query)
        conn.commit()
        
        # Verify row count
        cur.execute("SELECT COUNT(*) FROM ml_features_v2;")
        count = cur.fetchone()[0]
        print(f"Successfully created ml_features_v2 with {count} records!")
        
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    create_ml_view()
