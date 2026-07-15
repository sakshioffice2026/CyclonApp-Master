using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CyclonApp.Repositories.Repositories;

namespace CyclonApp.Repositories.Contracts
{
    public interface IUnitOfWork : IDisposable
    {
        ExceptionHandlerRepository exceptionHandlerRepository { get; }

        Task Commit();
    }
}
