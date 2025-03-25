-- Function to sync users from auth.users to public.users
CREATE OR REPLACE FUNCTION sync_auth_users()
RETURNS TRIGGER AS $$
BEGIN
    -- Insert into users table if user doesn't exist
    INSERT INTO public.users (user_id)
    VALUES (NEW.id)
    ON CONFLICT (user_id) DO NOTHING;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger on auth.users table
DROP TRIGGER IF EXISTS sync_users_trigger ON auth.users;
CREATE TRIGGER sync_users_trigger
    AFTER INSERT OR UPDATE
    ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION sync_auth_users();
