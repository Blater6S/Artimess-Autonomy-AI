# Author : P.P. Chanchal
import sys
import os

print("Starting Diagnosis...")

def check_import(module_name):
    try:
        print(f"Importing {module_name}...")
        __import__(module_name)
        print(f"SUCCESS: {module_name} imported.")
        return True
    except Exception as e:
        print(f"FAILURE: Could not import {module_name}. Error: {e}")
        return False

modules = ['AI_net', 'AI_data', 'AI_img', 'AI_txt', 'AI_voice']
results = {}

for mod in modules:
    results[mod] = check_import(mod)

print("\n--- Summary ---")
all_passed = True
for mod, success in results.items():
    status = "PASS" if success else "FAIL"
    print(f"{mod}: {status}")
    if not success:
        all_passed = False

if all_passed:
    print("\nAll modules imported successfully.")
    # Try initializing the coordinator
    try:
        print("\nAttempting to initialize AutonomousDataManager...")
        import AI_data
        mem = AI_data.VectorMemoryManager()
        mgr = AI_data.AutonomousDataManager(mem)
        print("SUCCESS: AutonomousDataManager initialized.")
    except Exception as e:
        print(f"FAILURE: AutonomousDataManager initialization failed. Error: {e}")
else:
    print("\nSome modules failed to import. Fix them first.")
