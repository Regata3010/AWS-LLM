echo "Scaling down BiasGuard..."
kubectl scale deployment biasguard-backend --replicas=0 -n biasguard
kubectl scale deployment biasguard-frontend --replicas=0 -n biasguard
kubectl scale deployment biasguard-redis --replicas=0 -n biasguard
kubectl scale deployment biasguard-postgres --replicas=0 -n biasguard
echo "BiasGuard scaled to zero"