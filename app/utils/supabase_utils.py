"""
Utility module for handling bulk Supabase operations.
This module provides functions for performing bulk inserts and updates to Supabase.
"""
import logging
from typing import List, Dict, Any, Optional, Union
from supabase import Client

# Initialize logging
logger = logging.getLogger(__name__)

def bulk_insert(
    supabase_client: Client,
    table: str,
    records: List[Dict[str, Any]],
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Insert multiple records into a Supabase table in batches.
    
    Args:
        supabase_client (Client): Supabase client instance
        table (str): Name of the table to insert into
        records (List[Dict[str, Any]]): List of records to insert
        batch_size (int): Number of records to insert in each batch
        
    Returns:
        Dict[str, Any]: Result of the operation with success count and errors
    """
    if not records:
        logger.warning("No records provided for bulk insert")
        return {"success_count": 0, "errors": []}
    
    result = {
        "success_count": 0,
        "errors": []
    }
    
    # Process records in batches
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            response = supabase_client.table(table).insert(batch).execute()
            
            if response.data:
                result["success_count"] += len(response.data)
                logger.info(f"Successfully inserted {len(response.data)} records into {table}")
            else:
                logger.warning(f"No data returned from insert operation for batch {i//batch_size + 1}")
                
        except Exception as e:
            error_msg = f"Error inserting batch {i//batch_size + 1}: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
    
    return result

def bulk_update(
    supabase_client: Client,
    table: str,
    records: List[Dict[str, Any]],
    id_field: str = "id",
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Update multiple records in a Supabase table in batches.
    
    Args:
        supabase_client (Client): Supabase client instance
        table (str): Name of the table to update
        records (List[Dict[str, Any]]): List of records to update
        id_field (str): Name of the ID field to use for matching records
        batch_size (int): Number of records to update in each batch
        
    Returns:
        Dict[str, Any]: Result of the operation with success count and errors
    """
    if not records:
        logger.warning("No records provided for bulk update")
        return {"success_count": 0, "errors": []}
    
    result = {
        "success_count": 0,
        "errors": []
    }
    
    # Process records in batches
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            # For each record in the batch, perform an update
            for record in batch:
                record_id = record.get(id_field)
                if not record_id:
                    error_msg = f"Record missing {id_field} field: {record}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)
                    continue
                
                # Remove the ID field from the update data
                update_data = {k: v for k, v in record.items() if k != id_field}
                
                response = supabase_client.table(table).update(update_data).eq(id_field, record_id).execute()
                
                if response.data:
                    result["success_count"] += len(response.data)
                else:
                    logger.warning(f"No data returned from update operation for record {record_id}")
                    
        except Exception as e:
            error_msg = f"Error updating batch {i//batch_size + 1}: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
    
    return result

def bulk_upsert(
    supabase_client: Client,
    table: str,
    records: List[Dict[str, Any]],
    id_field: str = "id",
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Insert or update multiple records in a Supabase table in batches.
    
    Args:
        supabase_client (Client): Supabase client instance
        table (str): Name of the table to upsert into
        records (List[Dict[str, Any]]): List of records to upsert
        id_field (str): Name of the ID field to use for matching records
        batch_size (int): Number of records to upsert in each batch
        
    Returns:
        Dict[str, Any]: Result of the operation with success count and errors
    """
    if not records:
        logger.warning("No records provided for bulk upsert")
        return {"success_count": 0, "errors": []}
    
    result = {
        "success_count": 0,
        "errors": []
    }
    
    # Process records in batches
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            response = supabase_client.table(table).upsert(batch).execute()
            
            if response.data:
                result["success_count"] += len(response.data)
                logger.info(f"Successfully upserted {len(response.data)} records into {table}")
            else:
                logger.warning(f"No data returned from upsert operation for batch {i//batch_size + 1}")
                
        except Exception as e:
            error_msg = f"Error upserting batch {i//batch_size + 1}: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
    
    return result