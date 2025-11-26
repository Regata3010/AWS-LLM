echo "Scaling up BiasGuard..."
kubectl scale deployment biasguard-postgres --replicas=1 -n biasguard
sleep 10  # Wait for DB to be ready
kubectl scale deployment biasguard-redis --replicas=1 -n biasguard
kubectl scale deployment biasguard-backend --replicas=2 -n biasguard
kubectl scale deployment biasguard-frontend --replicas=2 -n biasguard
echo "BiasGuard is live at http://35.202.6.191"